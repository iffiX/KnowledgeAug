#define FMT_HEADER_ONLY

#include "concept_net.h"
#include "tqdm.h"
#include "fmt/format.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xtensor.hpp"
#include "highfive/H5File.hpp"
#include "highfive/H5DataSpace.hpp"
#include <memory>
#include <regex>
#include <fstream>
#include <filesystem>

#define CONCPETNET_LINES 34074917
#define GLOVE_42B_300D_LINES 1917494

using namespace std;

string replaceAllChar(const string &input, char pattern, char replace) {
    string output(input.length(), ' ');
    for (size_t i = 0; i < input.length(); i++)
        output[i] = input[i] == pattern ? replace : input[i];
    return move(output);
}

vector<int> string2Vec(const string &input) {
    vector<int> vec;
    for (char c : input) {
        // prevent negative value when encontering non ASCII characters, since char is signed
        vec.push_back(int(c) + 1000);
    }
    return move(vec);
}

string vec2String(const vector<int> &input) {
    string result;
    for (int i : input)
        result +=(char)(i - 1000);
    return move(result);
}

KnowledgeBase
ConceptNetReader::read(const std::string &assertionPath, const std::string &weightPath, const std::string &weightStyle,
                       const std::string &weightHDF5Path, bool simplifyWithInt8) {
    KnowledgeBase kb;

    ifstream file(assertionPath);
    string line;
    unordered_map<string, long> relationshipsMap;
    unordered_set<string> rawRelationshipsSet;
    unordered_map<string, long> nodeMap;

    if (not weightPath.empty()) {
        if (weightHDF5Path.empty())
            throw std::invalid_argument("Weight path specified but hdf5 output path not specified.");
        readWeights(weightPath, weightStyle);
    }

    auto hdf5Directory = std::filesystem::path(weightHDF5Path).parent_path();
    if (not std::filesystem::exists(hdf5Directory)){
        auto ret = std::filesystem::create_directories(hdf5Directory);
        if (not ret)
            throw std::runtime_error("Failed to create parent directory for the HDF5 file.");
    }
    tqdm bar;
    bar.set_theme_basic();
    bar.disable_colors();
    size_t processed = 0;

    cout << "Begin processing ConceptNet." << endl;
    while (getline(file, line)) {
        istringstream is(line);
        string uri, relation, srcEntity, tarEntity, info;
        getline(is, uri, '\t');
        getline(is, relation, '\t');
        getline(is, srcEntity, '\t');
        getline(is, tarEntity, '\t');
        getline(is, info, '\t');
        size_t prefixLen = sizeof("/c/en/") - 1;
        if (srcEntity.length() >= prefixLen && srcEntity.substr(0, prefixLen) == "/c/en/" &&
            tarEntity.length() >= prefixLen && tarEntity.substr(0, prefixLen) == "/c/en/") {

            string src = replaceAllChar(srcEntity.substr(prefixLen, srcEntity.find("/", prefixLen) - prefixLen), '_',
                                        ' ');
            string tar = replaceAllChar(tarEntity.substr(prefixLen, tarEntity.find("/", prefixLen) - prefixLen), '_',
                                        ' ');
            long srcId = -1, tarId = -1;
            if (nodeMap.find(src) != nodeMap.end())
                srcId = nodeMap[src];
            else {
                long id = long(nodeMap.size());
                srcId = nodeMap[src] = id;
                kb.nodes.push_back(src);
            }
            if (nodeMap.find(tar) != nodeMap.end())
                tarId = nodeMap[tar];
            else {
                long id = long(nodeMap.size());
                tarId = nodeMap[tar] = id;
                kb.nodes.push_back(tar);
            }

            string rawRel = relation.substr(sizeof("/r/") - 1);
            if (rawRelationshipsSet.find(rawRel) == rawRelationshipsSet.end()) {
                rawRelationshipsSet.insert(rawRel);
                kb.rawRelationships.push_back(rawRel);
            }
            // only keep the last part of the relation
            // eg: for dbpedia/occupation it would be occupation
            auto pos = rawRel.rfind('/');
            string rel;
            long relId;
            if (pos == string::npos)
                rel = rawRel;
            else
                rel = rawRel.substr(pos + 1);
            if (relationshipsMap.find(rel) == relationshipsMap.end()) {
                long id = long(relationshipsMap.size());
                relId = relationshipsMap[rel] = id;
                kb.relationships.push_back(rel);
            } else
                relId = relationshipsMap[rel];

            float weight;
            string annotation;
            smatch result;
            if (regex_search(info, result, regex(R"("weight":\s+([+-]?([0-9]*[.])?[0-9]+))")))
                weight = stof(result[1]);
            else
                weight = 1;
            smatch result2;
            if (regex_search(info, result2, regex("\"surfaceText\":\\s+\"(.+)\"[,}]"))) {
                annotation = replaceAllChar(result2[1], '[', ' ');
                annotation = replaceAllChar(annotation, ']', ' ');
            }
            kb.edgeToTarget[tarId].push_back(kb.edges.size());
            kb.edgeFromSource[srcId].push_back(kb.edges.size());
            kb.edges.emplace_back(make_tuple(srcId, relId, tarId, weight, annotation));
        }
        processed++;
        bar.progress(processed, CONCPETNET_LINES);
    }
    bar.finish();
    cout << "ConceptNet statistics: " << endl;
    cout << "Node num: " << kb.nodes.size() << endl;
    cout << "Edge num: " << kb.edges.size() << endl;
    cout << "Relation num: " << kb.relationships.size() << endl;
    cout << "Raw relation num: " << kb.rawRelationships.size() << endl;

    if (not weightPath.empty()) {
        cout << "Begin associating ConceptNet Node to embedding." << endl;
        // Replace digits with # in any term where a sequence of two digits appears.
        // https://github.com/commonsense/conceptnet5/blob/master/conceptnet5/vectors/__init__.py
        auto doubleDigitRE = regex("[0-9][0-9]");
        auto digitRE = regex("[0-9]");

        auto nodeEmbeddingFile = std::make_unique<HighFive::File>(weightHDF5Path, HighFive::File::Overwrite);
        HighFive::DataSpace space({kb.nodes.size(), dim});
        HighFive::DataSpace nameSpace({kb.nodes.size()});
        HighFive::DataSetCreateProps createProps;
        HighFive::DataSetAccessProps accessProps;
        int elementSize = simplifyWithInt8 ? sizeof(int8_t) : sizeof(float);
        createProps.add(HighFive::Chunking({128, dim}));
        accessProps.add(HighFive::Caching(8192, 8192 * 128 * dim * elementSize));
        auto nameDataset = nodeEmbeddingFile->createDataSet<string>("names", nameSpace);
        auto dataset = simplifyWithInt8 ?
                       nodeEmbeddingFile->createDataSet<int8_t>("embeddings", space, createProps, accessProps) :
                       nodeEmbeddingFile->createDataSet<float>("embeddings", space, createProps, accessProps);
        size_t exactMatch = 0, mixedMatch = 0, missing = 0;

        bar.reset();
        for (size_t n = 0; n < kb.nodes.size(); n++) {
            auto &node = kb.nodes[n];
            nameDataset.select({n}, {1}).write(node);

            string cleaned;
            if (regex_search(node, doubleDigitRE))
                cleaned = regex_replace(node, digitRE, "#");
            else
                cleaned = node;

            if (weightNames.find(cleaned) != weightNames.end()) {
                // A perfect match is found
                size_t rawId = weightNames[cleaned];
                if (simplifyWithInt8) {
                    xt::xtensor<int8_t, 1> tmpResult = xt::cast<int8_t>(
                            xt::adapt(weights.get() + rawId * dim, dim, xt::no_ownership(),
                                      vector<size_t>{dim}) * 64);
                    dataset.select({n, 0}, {1, dim}).write(tmpResult.data());
                } else
                    dataset.select({n, 0}, {1, dim}).write(weights.get() + rawId * dim);
                exactMatch += 1;
            } else {
                // Find all matching raw weight names, use their average embedding
                // If there is no matching, set the first element to 1
                auto sentence = string2Vec(cleaned);
                auto result = weightNameTrie.matchForAll(sentence, false);
                xt::xtensor<float, 1> emb(xt::xtensor<float, 1>::shape_type{dim}, 0);
                for (auto &item : result) {
                    size_t rawId = weightNames.at(vec2String(item.second.back()));
                    emb += xt::adapt(weights.get() + rawId * dim, dim, xt::no_ownership(),
                                     vector<size_t>{dim});
                }
                if (not result.empty()) {
                    emb /= result.size();
                    mixedMatch += 1;
                } else {
                    emb[0] = 1;
                    missing += 1;
                }
                if (simplifyWithInt8) {
                    xt::xtensor<int8_t, 1> tmpResult = xt::cast<int8_t>(emb * 64);
                    dataset.select({n, 0}, {1, dim}).write_raw(tmpResult.data());
                } else
                    dataset.select({n, 0}, {1, dim}).write_raw(emb.data());
            }
            bar.progress(n, kb.nodes.size());
        }
        bar.finish();

        cout << fmt::format("Total num {}, exact match {:.1f}%, mixed match {:.1f}%, missing {:.1f}%",
                            kb.nodes.size(),
                            100 * float(exactMatch) / kb.nodes.size(),
                            100 * float(mixedMatch) / kb.nodes.size(),
                            100 * float(missing) / kb.nodes.size()) << endl;

        kb.setNodeEmbeddingFileName(weightHDF5Path);
        cout << "Finished" << endl;
    }
    return kb;
}

void ConceptNetReader::readWeights(const std::string &weightPath, const std::string &weightStyle) {
    ifstream file(weightPath);

    string line, numString, dimString;

    tqdm bar;
    bar.set_theme_basic();
    bar.disable_colors();

    if (not file.is_open())
        throw std::invalid_argument(fmt::format("Cannot open weight path {}", weightPath));

    if (weightStyle == "numberbatch") {
        getline(file, line);
        istringstream first(line);
        first >> numString >> dimString;
        size_t num = stoi(numString);
        dim = stoi(dimString);
        size_t processed = 0;

        cout << "Begin processing ConceptNet NumberBatch." << endl;
        weights = make_unique<float[]>(num * dim);
        while (getline(file, line)) {
            istringstream is(line);
            string name, weight;
            is >> name;
            size_t offset = processed * dim;
            for (size_t i = 0; i < dim; i++) {
                is >> weight;
                weights.get()[offset + i] = stof(weight);
            }
            name = replaceAllChar(name, '_', ' ');
            weightNames[name] = processed;
            weightNameTrie.insert(string2Vec(name));
            processed++;
            bar.progress(processed, num);
        }
    }
    else if (weightStyle.size() > 5 & weightStyle.compare(0, 5, "glove") == 0) {
        size_t num;
        if (weightStyle == "glove_42b_300d")
            num=GLOVE_42B_300D_LINES, dim = 300;
        else
            throw std::invalid_argument(fmt::format("Invalid GloVe weight style {}", weightStyle));
        size_t processed = 0;

        cout << "Begin processing GloVe." << endl;
        weights = make_unique<float[]>(num * dim);
        while (getline(file, line)) {
            istringstream is(line);
            string name, weight;
            is >> name;
            size_t offset = processed * dim;
            for (size_t i = 0; i < dim; i++) {
                is >> weight;
                weights.get()[offset + i] = stof(weight);
            }
            weightNames[name] = processed;
            weightNameTrie.insert(string2Vec(name));
            processed++;
            bar.progress(processed, num);
        }
    }
    else
        throw std::invalid_argument(fmt::format("Invalid weight style {}", weightStyle));
    bar.finish();
}
