#define FMT_HEADER_ONLY

#include "matcher.h"
#include "tqdm.h"
#include "fmt/format.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xtensor.hpp"
#include "highfive/H5File.hpp"
#include <stack>
#include <queue>
#include <tuple>
#include <random>
#include <memory>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string_view>

using namespace std;

bool isAllMasked(size_t start, size_t end, const vector<int> &mask) {
    bool allMasked = not mask.empty();
    if (not mask.empty()) {
        for (size_t i = start; i < end; i++) {
            if (mask[i] == 1) {
                allMasked = false;
                break;
            }
        }
    }
    return allMasked;
}

template <typename InIt1, typename InIt2>
bool unordered_set_has_intersection(InIt1 b1, InIt1 e1, InIt2 b2, InIt2 e2) {
    // For performance, put smaller set at first
    while (!(b1 == e1)) {
        if (!(find(b2, e2, *b1) == e2)) {
            return true;
        }
        ++b1;
    }
    return false;
}

template<typename T>
UnorderedPair<T>::UnorderedPair(T value1, T value2) : value1(value1), value2(value2) {}

template<typename T>
bool UnorderedPair<T>::operator==(const UnorderedPair<T> &other) {
    return (value1 == other.value1 && value2 == other.value2) ||
           (value2 == other.value1 && value1 == other.value2);
}

template<typename T>
size_t UnorderedPairHash<T>::operator()(const UnorderedPair<T> &pair) const {
    return hash<T>(pair.value1) ^ hash<T>(pair.value2);
}

TrieNode::TrieNode(int token, bool isWordEnd) : token(token), isWordEnd(isWordEnd) {}

TrieNode::~TrieNode() {
    clear();
}

TrieNode *TrieNode::addChild(int childToken, bool isChildWordEnd) {
    children[childToken] = new TrieNode(childToken, isChildWordEnd);
    return children[childToken];
}

bool TrieNode::removeChild(int childToken) {
    if (children.find(childToken) != children.end()) {
        delete children[childToken];
        children.erase(childToken);
        return true;
    } else
        return false;
}

void TrieNode::clear() {
    for (auto &child : children)
        delete child.second;
    children.clear();
}

Trie::Trie(const vector<vector<int>> &words) {
    for (auto &word : words)
        insert(word);
}

void Trie::insert(const vector<int> &word) {
    auto *currentNode = &root;
    for (int token : word) {
        if (token < 0)
            throw invalid_argument("Word must be a positive integer.");
        auto child = currentNode->children.find(token);
        if (child != currentNode->children.end())
            currentNode = child->second;
        else
            currentNode = currentNode->addChild(token, false);
    }
    currentNode->isWordEnd = true;
}

bool Trie::remove(const vector<int> &word) {
    TrieNode *parentNode = nullptr;
    auto *currentNode = &root;
    if (word.empty())
        return false;
    for (int token : word) {
        auto child = currentNode->children.find(token);
        if (child != currentNode->children.end()) {
            parentNode = currentNode;
            currentNode = child->second;
        } else
            return false;
    }
    if (currentNode->isWordEnd) {
        if (currentNode->children.empty())
            parentNode->removeChild(currentNode->token);
        else
            currentNode->isWordEnd = false;
        return true;
    } else
        return false;
}

void Trie::clear() {
    root.clear();
}

vector<vector<int>> Trie::matchForStart(const vector<int> &sentence, size_t start, bool allowSubMatch) const {
    vector<int> tmp;
    vector<vector<int>> result;
    auto *currentNode = &root;
    for (size_t i = start; i < sentence.size(); i++) {
        auto child = currentNode->children.find(sentence[i]);
        if (child != currentNode->children.end()) {
            currentNode = child->second;
            tmp.push_back(currentNode->token);
            if (currentNode->isWordEnd) {
                // move temporary memory to result
                result.push_back(tmp);
            }
        } else
            break;
    }
#ifdef DEBUG
    cout << "Matched results:" << endl;
    for (auto &r : result)
        cout << fmt::format("[{}]", fmt::join(r.begin(), r.end(), ",")) << endl;
#endif
    if (not allowSubMatch and not result.empty())
        result.erase(result.begin(), result.begin() + result.size() - 1);
    return move(result);
}

unordered_map<size_t, vector<vector<int>>> Trie::matchForAll(const vector<int> &sentence, bool allowSubMatch) const {
    unordered_map<size_t, vector<vector<int>>> result;
    for (size_t i = 0; i < sentence.size();) {
        vector<vector<int>> matches = move(matchForStart(sentence, i, allowSubMatch));
        if (not matches.empty()) {
            size_t match_size = matches.front().size();
            result.emplace(i, matches);
            i += match_size;
        } else
            i++;
    }
    return move(result);
}

void KnowledgeBase::clearDisabledEdges() {
    isEdgeDisabled.clear();
    isEdgeDisabled.resize(edges.size(), false);
}

void KnowledgeBase::disableAllEdges() {
    isEdgeDisabled.clear();
    isEdgeDisabled.resize(edges.size(), true);
}

void KnowledgeBase::disableEdgesWithWeightBelow(float minWeight) {
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (get<3>(edges[edgeIndex]) < minWeight)
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::disableEdgesOfRelationships(const vector<string> &rel, bool excludeCompositeNodes) {
    unordered_set<string> disabledSet(rel.begin(), rel.end());
    unordered_set<long> disabledIds;
    for (long relationshipId = 0; relationshipId < relationships.size(); relationshipId++) {
        if (disabledSet.find(relationships[relationshipId]) != disabledSet.end())
            disabledIds.insert(relationshipId);
    }
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        bool skip = excludeCompositeNodes and
                    (isNodeComposite[get<0>(edges[edgeIndex])] or isNodeComposite[get<2>(edges[edgeIndex])]);
        if (not skip and disabledIds.find(get<1>(edges[edgeIndex])) != disabledIds.end())
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::enableEdgesOfRelationships(const vector<string> &rel, bool excludeCompositeNodes) {
    unordered_set<string> enabledSet(rel.begin(), rel.end());
    unordered_set<long> enabledIds;
    for (long relationshipId = 0; relationshipId < relationships.size(); relationshipId++) {
        if (enabledSet.find(relationships[relationshipId]) != enabledSet.end())
            enabledIds.insert(relationshipId);
    }
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        bool skip = excludeCompositeNodes and
                    (isNodeComposite[get<0>(edges[edgeIndex])] or isNodeComposite[get<2>(edges[edgeIndex])]);
        if (not skip and enabledIds.find(get<1>(edges[edgeIndex])) == enabledIds.end())
            isEdgeDisabled[edgeIndex] = true;
    }
}

void KnowledgeBase::disableEdgesOfNodes(const vector<string> &nod) {
    unordered_set<string> disabledSet(nod.begin(), nod.end());
    unordered_set<long> disabledIds;
    for (long nodeId = 0; nodeId < nodes.size(); nodeId++) {
        if (disabledSet.find(nodes[nodeId]) != disabledSet.end()) {
#ifdef DEBUG
            cout << fmt::format("Found node to be disabled [{}:{}]", nodes[nodeId], nodeId) << endl;
#endif
            disabledIds.insert(nodeId);
        }
    }
#ifdef DEBUG
    cout << "Begin disabling edges" << endl;
#endif
    for (size_t edgeIndex = 0; edgeIndex < edges.size(); edgeIndex++) {
        if (disabledIds.find(get<0>(edges[edgeIndex])) != disabledIds.end() ||
            disabledIds.find(get<2>(edges[edgeIndex])) != disabledIds.end()) {
#ifdef DEBUG
            cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}]",
                                edgeIndex,
                                nodes[get<0>(edges[edgeIndex])], get<0>(edges[edgeIndex]),
                                relationships[get<1>(edges[edgeIndex])], get<1>(edges[edgeIndex]),
                                nodes[get<2>(edges[edgeIndex])], get<2>(edges[edgeIndex])) << endl;
#endif
            isEdgeDisabled[edgeIndex] = true;
        }
    }
#ifdef DEBUG
    cout << "End disabling edges" << endl;
#endif
}

vector<long> KnowledgeBase::findNodes(const vector<string> &nod, bool quiet) const {
    vector<long> ids;
    for (auto &nName : nod) {
        bool found = nodeToIndex.find(nName) != nodeToIndex.end();
        if (found)
            ids.push_back(nodeToIndex.at(nName));
        else {
            if (not quiet)
                throw invalid_argument(fmt::format("Node {} not found", nName));
            else
                ids.push_back(-1);
        }

    }
    return move(ids);
}

unordered_set<long> KnowledgeBase::getNodeNeighbors(long node) const {
    if (adjacency.find(node) == adjacency.end())
        throw invalid_argument("Node doesn't exist");
    return move(adjacency.at(node));
}

vector<Edge> KnowledgeBase::getEdges(long source, long target) const {
    vector<Edge> result;
    if (source != -1 && target == -1) {
        if (edgeFromSource.find(source) != edgeFromSource.end()) {
            for (size_t edgeIndex: edgeFromSource.at(source))
                if (not isEdgeDisabled[edgeIndex])
                    result.push_back(edges[edgeIndex]);
        }

    } else if (source == -1 && target != -1) {
        if (edgeToTarget.find(target) != edgeToTarget.end()) {
            for (size_t edgeIndex: edgeToTarget.at(target))
                if (not isEdgeDisabled[edgeIndex])
                    result.push_back(edges[edgeIndex]);
        }
    } else if (edgeFromSource.find(source) != edgeFromSource.end()) {
        for (size_t edgeIndex : edgeFromSource.at(source)) {
            if (not isEdgeDisabled[edgeIndex] and get<2>(edges[edgeIndex]) == target)
                result.push_back(edges[edgeIndex]);
        }
    }

    return move(result);
}

vector<Edge> KnowledgeBase::getEdgesBidirection(long node1, long node2) const {
    vector<Edge> result;
    if (edgeFromSource.find(node1) != edgeFromSource.end()) {
        for (size_t edgeIndex : edgeFromSource.at(node1))
            if (not isEdgeDisabled[edgeIndex] and get<2>(edges[edgeIndex]) == node2)
                result.push_back(edges[edgeIndex]);
    }
    if (edgeFromSource.find(node2) != edgeFromSource.end()) {
        for (size_t edgeIndex : edgeFromSource.at(node2))
            if (not isEdgeDisabled[edgeIndex] and get<2>(edges[edgeIndex]) == node1)
                result.push_back(edges[edgeIndex]);
    }
    return move(result);
}

const vector<string> &KnowledgeBase::getNodes() const {
    return nodes;
}

vector<string> KnowledgeBase::getNodes(const vector<long> &nodeIndexes) const {
    vector<string> result;
    for (long nodeIdx : nodeIndexes) {
        result.push_back(nodes[nodeIdx]);
    }
    return result;
}

void KnowledgeBase::addCompositeNode(const string &compositeNode,
                                     const string &relationship,
                                     const vector<int> &tokenizedCompositeNode,
                                     const vector<int> &mask,
                                     const vector<int> &connectionMask,
                                     size_t splitNodeMinimumEdgeNum,
                                     float splitNodeMinimumSimilarity) {
    // Note: The similaity of the composite node to other nodes is computed by:
    // the maximum sub node similarity to other nodes

    long relationId = -1;
    for (long relId = 0; relId < long(relationships.size()); relId++) {
        if (relationships[relId] == relationship) {
            relationId = relId;
            break;
        }
    }
    if (relationId == -1)
        throw invalid_argument(fmt::format("Relationship [{}] not found", relationship));

    long newNodeId = long(nodes.size());
    nodes.emplace_back(compositeNode);
    tokenizedNodes.emplace_back(tokenizedCompositeNode);
    isNodeComposite.push_back(true);
    // Do not update nodeMap and nodeTrie because it is a composite node

    // Find all sub-nodes occuring in the composite node
    // Then add an edge from all sub nodes to the composite node with relationship=relationship

    if (not mask.empty() && mask.size() != tokenizedCompositeNode.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for composite node but size does not match, composite node: {}, mask: {}",
                tokenizedCompositeNode.size(), mask.size()));

    if (not connectionMask.empty() && connectionMask.size() != tokenizedCompositeNode.size())
        throw invalid_argument(fmt::format(
                "Connection mask is provided for composite node but size does not match, composite node: {}, connection mask: {}",
                tokenizedCompositeNode.size(), connectionMask.size()));

    auto result = nodeTrie.matchForAll(tokenizedCompositeNode, false);
    unordered_map<long, float> components;
    unordered_set<long> connectedSource;
    for (auto &subNode : result) {
        size_t subNodeSize = subNode.second.back().size();
        if (isAllMasked(subNode.first, subNode.first + subNodeSize, mask))
            continue;

        long subNodeId = nodeMap.at(subNode.second.back());
        bool hasOut = edgeFromSource.find(subNodeId) != edgeFromSource.end();
        bool hasIn = edgeToTarget.find(subNodeId) != edgeToTarget.end();
        size_t outSize = hasOut ? edgeFromSource.at(subNodeId).size() : 0;
        size_t inSize = hasIn ? edgeToTarget.at(subNodeId).size() : 0;

        if (outSize + inSize < splitNodeMinimumEdgeNum) {
#ifdef DEBUG
            cout << fmt::format("Splitting node [{}:{}]", nodes[subNodeId], subNodeId) << endl;
#endif
            unordered_map<size_t, vector<vector<int>>> subSubMatches = nodeTrie.matchForAll(
                    subNode.second.back(), true);

            for (auto &subSubMatch : subSubMatches) {
                // See normalize match
                for (auto &baseMatch : subSubMatch.second) {
                    if (not isAllMasked(subSubMatch.first + subNode.first,
                                        subSubMatch.first + subNode.first + baseMatch.size(), mask)) {
                        long baseNodeId = nodeMap.at(baseMatch);
                        if (cosineSimilarity(baseNodeId, subNodeId) > splitNodeMinimumSimilarity) {
                            if (connectedSource.find(baseNodeId) == connectedSource.end()) {
                                connectedSource.insert(baseNodeId);
                                if (not isAllMasked(subSubMatch.first + subNode.first,
                                                    subSubMatch.first + subNode.first + baseMatch.size(), connectionMask)) {
                                    addCompositeEdge(baseNodeId, relationId, newNodeId);
                                }
                                components[baseNodeId] += 1;
                                compositeComponentCount[baseNodeId] += 1;
                            }

#ifdef DEBUG
                            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
                                                nodes[baseNodeId], baseNodeId,
                                                nodes[newNodeId], newNodeId) << endl;
#endif
                        }
                    }
                }
            }
        } else {
            if (connectedSource.find(subNodeId) == connectedSource.end()) {
                connectedSource.insert(subNodeId);
                if (not isAllMasked(subNode.first, subNode.first + subNodeSize, connectionMask)) {
                    addCompositeEdge(subNodeId, relationId, newNodeId);
                }
                components[subNodeId] += 1;
                compositeComponentCount[subNodeId] += 1;
            }

#ifdef DEBUG
            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
                                nodes[subNodeId], subNodeId,
                                nodes[newNodeId], newNodeId) << endl;
#endif
        }
    }
    compositeNodes.emplace(newNodeId, components);
    nodeToIndex[compositeNode] = newNodeId;
}

void KnowledgeBase::addCompositeEdge(long sourceNodeId, long relationId, long compositeNodeId) {
    if (sourceNodeId < 0 || sourceNodeId >= nodes.size())
        throw invalid_argument(fmt::format("Invalid source node {}", sourceNodeId));
    if (compositeNodeId < 0 || compositeNodeId >= nodes.size() || not isNodeComposite[compositeNodeId])
        throw invalid_argument(fmt::format("Invalid target node {}", compositeNodeId));
    size_t edgeIndex = edges.size();
    edges.emplace_back(Edge{sourceNodeId, relationId, compositeNodeId, 1, ""});
    edgeToTarget[compositeNodeId].push_back(edgeIndex);
    edgeFromSource[sourceNodeId].push_back(edgeIndex);
    isEdgeDisabled.push_back(false);
    adjacency[compositeNodeId].insert(sourceNodeId);
    adjacency[sourceNodeId].insert(compositeNodeId);
    tokenizedEdgeAnnotations.emplace_back(vector<int>{});
#ifdef DEBUG
    cout << fmt::format("Connecting node [{}:{}] to composite node [{}:{}] with relation [{}:{}]",
                        nodes[sourceNodeId], sourceNodeId,
                        nodes[compositeNodeId], compositeNodeId,
                        relationships[relationId], relationId) << endl;
#endif
}

void KnowledgeBase::setNodeEmbeddingFileName(const string &path) {
    nodeEmbeddingFileName = path;
    loadEmbedding();
}

bool KnowledgeBase::isNeighbor(long node1, long node2) const {
    return adjacency.find(node1) != adjacency.end() && adjacency.at(node1).find(node2) != adjacency.at(node1).end();
}

int KnowledgeBase::bfsDistance(long node1, long node2, int maxDepth) const {
    vector<long> current = {node1};
    vector<bool> added;
    int depth = 0;

    // Note: BFS distance here treats the graph as not directional
    while (not current.empty()) {
        vector<long> next;
        added.clear();
        added.resize(nodes.size(), false);
        for (auto currentNode : current) {
            if (currentNode == node2)
                return depth;
            // Iterate all neighbors
            if (adjacency.find(currentNode) != adjacency.end()) {
                for (long nextNode : adjacency.at(currentNode)) {
                    if (not added[nextNode]) {
                        next.push_back(nextNode);
                        added[nextNode] = true;
                    }
                }
            }
        }
        current.swap(next);
        depth++;
        if (depth >= maxDepth)
            break;
    }
    return -1;
}

float KnowledgeBase::cosineSimilarity(long node1, long node2) const {
    if (isNodeComposite[node1] or isNodeComposite[node2])
        throw invalid_argument("Composite nodes are not supported");

    if (nodeEmbedding == nullptr)
        throw runtime_error("Embedding not configured");

    if (nodeEmbeddingSimplifyWithInt8) {
        const int8_t *emb = static_pointer_cast<int8_t[]>(nodeEmbedding).get();
        auto srcEmbed = xt::cast<int16_t>(xt::adapt(emb + node1 * nodeEmbeddingDim,
                                                       nodeEmbeddingDim,
                                                       xt::no_ownership(),
                                                    vector<size_t>{nodeEmbeddingDim}));
        auto tarEmbed = xt::cast<int16_t>(xt::adapt(emb + node2 * nodeEmbeddingDim,
                                                       nodeEmbeddingDim,
                                                       xt::no_ownership(),
                                                    vector<size_t>{nodeEmbeddingDim}));
        // cosine similarity
        float dot = xt::sum<int16_t>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(srcEmbed * srcEmbed)))[0];
        float norm2 = xt::sqrt(xt::cast<float>(xt::sum<int16_t>(tarEmbed * tarEmbed)))[0];
        return dot / (norm1 * norm2);
    } else {
        float *emb = static_pointer_cast<float[]>(nodeEmbedding).get();
        auto srcEmbed = xt::adapt(emb + node1 * nodeEmbeddingDim,
                                  nodeEmbeddingDim,
                                  xt::no_ownership(),
                                  vector<size_t>{nodeEmbeddingDim});
        auto tarEmbed = xt::adapt(emb + node2 * nodeEmbeddingDim,
                                  nodeEmbeddingDim,
                                  xt::no_ownership(),
                                  vector<size_t>{nodeEmbeddingDim});
        // cosine similarity
        float dot = xt::sum<float>(srcEmbed * tarEmbed)[0];
        float norm1 = xt::sqrt(xt::sum<float>(srcEmbed * srcEmbed))[0];
        float norm2 = xt::sqrt(xt::sum<float>(tarEmbed * tarEmbed))[0];
        return dot / (norm1 * norm2);
    }
}

void KnowledgeBase::save(const string &archivePath) const {
    KnowledgeArchive archive;
    if (not compositeNodes.empty())
        throw runtime_error("It's not safe to save after adding composite nodes");

    cout << "[KB] Begin saving" << endl;
    for (auto &ett : edgeToTarget)
        archive.edgeToTarget.emplace(ett.first, vector2ArchiveVector(ett.second));
    for (auto &efs : edgeFromSource)
        archive.edgeFromSource.emplace(efs.first, vector2ArchiveVector(efs.second));
    for (auto &e: edges)
        archive.edges.emplace_back(SerializableEdge{
                get<0>(e), get<1>(e), get<2>(e), get<3>(e), get<4>(e)});
    archive.nodes.set(nodes.begin(), nodes.end());
    archive.relationships.set(relationships.begin(), relationships.end());
    for (auto &tn : tokenizedNodes)
        archive.tokenizedNodes.emplace_back(vector2ArchiveVector(tn));
    for (auto &tr : tokenizedRelationships)
        archive.tokenizedRelationships.emplace_back(vector2ArchiveVector(tr));
    for (auto &tea : tokenizedEdgeAnnotations)
        archive.tokenizedEdgeAnnotations.emplace_back(vector2ArchiveVector(tea));
    archive.rawRelationships.set(rawRelationships.begin(), rawRelationships.end());
    archive.nodeEmbeddingFileName = nodeEmbeddingFileName;
    auto file = cista::file(archivePath.c_str(), "w");
    cista::serialize(file, archive);

    cout << "[KB] Saved node num: " << nodes.size() << endl;
    cout << "[KB] Saved edge num: " << edges.size() << endl;
    cout << "[KB] Saved relation num: " << relationships.size() << endl;
    cout << "[KB] Saved raw relation num: " << rawRelationships.size() << endl;
}

void KnowledgeBase::load(const string &archivePath) {
    edgeToTarget.clear();
    edgeFromSource.clear();
    edges.clear();
    isEdgeDisabled.clear();

    relationships.clear();
    rawRelationships.clear();

    nodes.clear();
    nodeTrie.clear();
    nodeMap.clear();
    isNodeComposite.clear();
    compositeNodes.clear();

    tokenizedNodes.clear();
    tokenizedRelationships.clear();
    tokenizedEdgeAnnotations.clear();

    nodeEmbedding = nullptr;

    cout << "[KB] Begin loading" << endl;
    auto file = cista::file(archivePath.c_str(), "r");
    auto content = file.content();
    auto *archive = cista::deserialize<KnowledgeArchive>(content);
    for (auto &ett : archive->edgeToTarget)
        edgeToTarget[ett.first] = archiveVector2Vector(ett.second);
    for (auto &efs : archive->edgeFromSource)
        edgeFromSource[efs.first] = archiveVector2Vector(efs.second);
    for (auto &e: archive->edges)
        edges.emplace_back(make_tuple(e.source, e.relation, e.target, e.weight, e.annotation));
    nodes.insert(nodes.end(), archive->nodes.begin(), archive->nodes.end());
    relationships.insert(relationships.end(), archive->relationships.begin(), archive->relationships.end());
    for (auto &tn : archive->tokenizedNodes)
        tokenizedNodes.emplace_back(archiveVector2Vector(tn));
    for (auto &tr : archive->tokenizedRelationships)
        tokenizedRelationships.emplace_back(archiveVector2Vector(tr));
    for (auto &tea : archive->tokenizedEdgeAnnotations)
        tokenizedEdgeAnnotations.emplace_back(archiveVector2Vector(tea));
    rawRelationships.insert(rawRelationships.end(), archive->rawRelationships.begin(), archive->rawRelationships.end());
    for (long index = 0; index < tokenizedNodes.size(); index++) {
        nodeTrie.insert(tokenizedNodes[index]);
        nodeMap[tokenizedNodes[index]] = index;
    }
    isNodeComposite.resize(nodes.size(), false);
    isEdgeDisabled.resize(edges.size(), false);
    nodeEmbeddingFileName = archive->nodeEmbeddingFileName;
    refresh();
    cout << "[KB] Loaded node num: " << nodes.size() << endl;
    cout << "[KB] Loaded edge num: " << edges.size() << endl;
    cout << "[KB] Loaded relation num: " << relationships.size() << endl;
    cout << "[KB] Loaded raw relation num: " << rawRelationships.size() << endl;
    cout << "[KB] Loading finished" << endl;
}

void KnowledgeBase::refresh() {
    loadEmbedding();
    loadAdjacency();
    loadNodeToIndex();
}

template<typename T1, typename T2>
bool KnowledgeBase::PriorityCmp::operator()(const pair<T1, T2> &pair1, const pair<T1, T2> &pair2) {
    return pair1.first > pair2.first;
}

size_t KnowledgeBase::VectorHash::operator()(const vector<int> &vec) const {
    // A simple hash function
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    size_t seed = vec.size();
    for (auto &i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void KnowledgeBase::loadEmbedding() {
    shared_ptr<HighFive::File> nodeEmbeddingFile;
    shared_ptr<HighFive::DataSet> nodeEmbeddingDataset;

    nodeEmbedding.reset();
    if (not nodeEmbeddingFileName.empty()) {
        ifstream tmpFile(nodeEmbeddingFileName);
        cout << "[KB] Embedding configured, using embedding file [" << nodeEmbeddingFileName << "]" << endl;
        if (tmpFile.fail())
            cout << "[KB] Failed to load embedding file [" << nodeEmbeddingFileName << "], skipped" << endl;
        else {
            tmpFile.close();
            nodeEmbeddingFile = make_shared<HighFive::File>(nodeEmbeddingFileName, HighFive::File::ReadOnly);
            nodeEmbeddingDataset = make_shared<HighFive::DataSet>(nodeEmbeddingFile->getDataSet("embeddings"));
            auto shape = nodeEmbeddingDataset->getDimensions();
            auto dtype = nodeEmbeddingDataset->getDataType();
            if (shape.size() != 2)
                throw invalid_argument(
                        fmt::format("Knowledge base embedding should be 2-dimensional, but got shape [{}]",
                                    fmt::join(shape.begin(), shape.end(), ",")));
            nodeEmbeddingDim = shape[1];
            nodeEmbeddingSimplifyWithInt8 = dtype.getClass() == HighFive::DataTypeClass::Integer;


            cout << "[KB] Loading embedding to memory" << endl;
            if (nodeEmbeddingSimplifyWithInt8) {
                nodeEmbedding = shared_ptr<int8_t[]>(new int8_t[shape[0] * shape[1]]);
                nodeEmbeddingDataset->read(static_pointer_cast<int8_t[]>(nodeEmbedding).get());
            } else {
                nodeEmbedding = shared_ptr<float[]>(new float[shape[0] * shape[1]]);
                nodeEmbeddingDataset->read(static_pointer_cast<float[]>(nodeEmbedding).get());
            }
            cout << "[KB] Closing embedding file" << endl;
            nodeEmbeddingDataset.reset();
            nodeEmbeddingFile.reset();

        }
    } else {
        cout << "[KB] Embedding not configured, skipped" << endl;
    }
}

void KnowledgeBase::loadAdjacency() {
    adjacency.clear();
    for (auto &srcNodeAdj : edgeFromSource) {
        for (auto edgeIndex : srcNodeAdj.second) {
            adjacency[srcNodeAdj.first].insert(get<2>(edges[edgeIndex]));
        }
    }
    for (auto &tarNodeAdj : edgeToTarget) {
        for (auto edgeIndex : tarNodeAdj.second) {
            adjacency[tarNodeAdj.first].insert(get<0>(edges[edgeIndex]));
        }
    }
}

void KnowledgeBase::loadNodeToIndex() {
    nodeToIndex.clear();
    for (long nodeId = 0; nodeId < nodes.size(); nodeId++) {
        nodeToIndex[nodes[nodeId]] = nodeId;
    }
}

template<typename T>
cista::raw::vector<T> KnowledgeBase::vector2ArchiveVector(const vector<T> &vec) {
    return move(cista::raw::vector<T>(vec.begin(), vec.end()));
}

template<typename T>
vector<T> KnowledgeBase::archiveVector2Vector(const cista::raw::vector<T> &vec) {
    return move(vector<T>(vec.begin(), vec.end()));
}

KnowledgeMatcher::KnowledgeMatcher(const KnowledgeBase &knowledgeBase) {
    cout << "[KM] Initializing matcher from knowledge base" << endl;
    kb = knowledgeBase;
    cout << "[KM] Matcher initialized" << endl;
}

KnowledgeMatcher::KnowledgeMatcher(const string &archivePath) {
    cout << "[KM] Initializing matcher from knowledge base archive" << endl;
    load(archivePath);
    cout << "[KM] Matcher initialized" << endl;
}

void KnowledgeMatcher::setCorpus(const vector<vector<int>> &corpus) {
    isCorpusSet = true;
    documentCountOfNodeInCorpus.clear();
    corpusSize = corpus.size();
    tqdm bar;
    bar.set_theme_basic();
    bar.disable_colors();
    cout << "Begin processing corpus." << endl;
    size_t processed = 0;
    for (auto &document : corpus) {
        auto result = kb.nodeTrie.matchForAll(document, false);
        unordered_set<long> insertedSubNodes;
        for (auto &subNode : result) {
            long subNodeId = kb.nodeMap.at(subNode.second.back());
            documentCountOfNodeInCorpus[subNodeId] += 1;
        }
        processed++;
        bar.progress(processed, corpus.size());
    }
    bar.finish();
}

string KnowledgeMatcher::findClosestConcept(string targetConcept, const vector<string> &concepts) {
    auto targetIdVec = kb.findNodes({targetConcept});
    long targetId = targetIdVec[0];
    vector<long> conceptIds = kb.findNodes(concepts, true);
    size_t bestConceptIdx = 0;
    float bestSimilarity = -1;
    for(size_t i = 0; i < conceptIds.size(); i++) {
        if (conceptIds[i] != -1) {
            float similarity = kb.cosineSimilarity(targetId, conceptIds[i]);
            if (similarity > bestSimilarity) {
                bestConceptIdx = i;
                bestSimilarity = similarity;
            }
        }
    }
    return concepts[bestConceptIdx];
}

KnowledgeMatcher::PathResult
KnowledgeMatcher::findShortestPath(const vector<int> &sourceSentence,
                                   const vector<int> &targetSentence,
                                   const vector<string> &intermediateNodes,
                                   const vector<int> &sourceMask,
                                   const vector<int> &targetMask,
                                   int maxDepthForEachNode,
                                   size_t splitNodeMinimumEdgeNum,
                                   float splitNodeMinimumSimilarity) {
#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: findShortestPath" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("intermediateNodes: [{}]",
                        fmt::join(intermediateNodes.begin(), intermediateNodes.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "maxDepthForEachNode: " << maxDepthForEachNode << endl;
    cout << "================================================================================" << endl;
#endif
    // start token position of the node, tokens made up of the node
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch,
                            splitNodeMinimumEdgeNum,
                            splitNodeMinimumSimilarity);

    if (sourceMatch.empty()) {
#ifdef DEBUG
        cout << "Source match result is empty, return" << endl;
#endif
        return move(PathResult());
    }
    if (targetMatch.empty()) {
#ifdef DEBUG
        cout << "Target match result is empty, return" << endl;
#endif
        return move(PathResult());
    }

    // node ids in sourceSentence / targetSentence, their occurence times
    unordered_map<long, float> sourceNodeOccurrences;
    unordered_map<long, float> targetNodeOccurrences;

    PathResult result;

    for (auto &sm : sourceMatch)
        sourceNodeOccurrences[kb.nodeMap.at(sm.second)] += 1;

    for (auto &tm : targetMatch)
        targetNodeOccurrences[kb.nodeMap.at(tm.second)] += 1;

    float compositeNodesCountSum = 0;
    float corpusNodesCountSum = 0;
    for (auto &count : documentCountOfNodeInCorpus)
        corpusNodesCountSum += count.second;
    for (auto &count : kb.compositeComponentCount)
        compositeNodesCountSum += count.second;

    // Construct search targets for each step
    vector<unordered_set<long>> searchIds;
    vector<long> sourceNodes, targetNodes;
    for (auto &sOccur : sourceNodeOccurrences)
        sourceNodes.push_back(sOccur.first);
    for (auto &tOccur : targetNodeOccurrences)
        targetNodes.push_back(tOccur.first);

    searchIds.emplace_back(unordered_set<long>(sourceNodes.begin(), sourceNodes.end()));
    for (auto &interNode : intermediateNodes)
        searchIds.emplace_back(unordered_set<long>{kb.findNodes({interNode})[0]});
    searchIds.emplace_back(unordered_set<long>(targetNodes.begin(), targetNodes.end()));

#ifdef DEBUG_DECISION
    cout << "SearchIds:" << endl;
    for (size_t i = 0; i < searchIds.size(); i++) {
        cout << fmt::format("Level {}: ", i);
        for (long node : searchIds[i])
            cout << fmt::format("({}) {}, ", node, kb.getNodes({node})[0]);
        cout << endl;
    }
#endif

    vector<Edge> bestPath;
    for (size_t level = 0; level < searchIds.size() - 1; level++)
    {
        vector<pair<vector<Edge>, float>> paths;

        size_t bestLevelDepth = maxDepthForEachNode;

        // rank start nodes by their similarity to the next level nodes
        vector<long> startNodes;
        if (level == 0) {
            vector<pair<long, float>> rank;
            long nextNode = *searchIds[1].begin();
            for (long startNode : searchIds[0]) {
                if (not kb.isNodeComposite[nextNode]) {
                    rank.push_back(make_pair(startNode, kb.cosineSimilarity(startNode, nextNode)));
                }
                else {
                    float bestSimilarity = -1;
                    for (auto component : kb.compositeNodes[nextNode]) {
                        float similarity = kb.cosineSimilarity(startNode, component.first);
                        if (similarity > bestSimilarity)
                            bestSimilarity = similarity;
                    }
                    rank.push_back(make_pair(startNode, bestSimilarity));
                }
            }
            sort(rank.begin(), rank.end(), [](const pair<long, float> &a, const pair<long, float> &b){
                return a.second > b.second;
            });
            for (auto &node : rank)
                startNodes.push_back(node.first);
        }
        else {
            startNodes.insert(startNodes.end(), searchIds[level].begin(), searchIds[level].end());
        }

        for (long startNode : startNodes) {
            vector<bool> visitedNodes(kb.nodes.size(), false);
            unordered_set<long> lastStepVisitedNodes;
            unordered_map<long, vector<long>> previousNodes;
            visitedNodes[startNode] = true;
            lastStepVisitedNodes.insert(startNode);
            previousNodes[startNode] = {};

            bool found = false;
            size_t step;
            for (step = 0; step < bestLevelDepth; step++) {
                unordered_set<long> newVisitedNodes;
                for (long lastNode : lastStepVisitedNodes) {
                    auto neighbors = kb.getNodeNeighbors(lastNode);
                    for (long neighbor : neighbors) {
                        // If the neighbor node is only visited in the current step
                        if (not visitedNodes[neighbor]) {
                            previousNodes[neighbor].push_back(lastNode);
                            newVisitedNodes.insert(neighbor);
                        }
                    }
                }
#ifdef DEBUG_DECISION
                cout << fmt::format("Step {} visited nodes {}", step, newVisitedNodes.size()) << endl;
#endif
                for (long newVisitedNode : newVisitedNodes)
                    visitedNodes[newVisitedNode] = true;
                found = unordered_set_has_intersection(searchIds[level + 1].begin(),
                                                       searchIds[level + 1].end(),
                                                       newVisitedNodes.begin(),
                                                       newVisitedNodes.end());
                lastStepVisitedNodes.swap(newVisitedNodes);
                if (found)
                    break;
            }
            if (found) {
                // Update max allowed depth
                bestLevelDepth = step + 1;
                // construct paths in the reverse order
                stack<pair<long, vector<long>>> unvisited;
                vector<vector<long>> reversedPathNodes;
                for(long node : searchIds[level + 1]) {
                    if (previousNodes.find(node) != previousNodes.end())
                        unvisited.push(make_pair(node, vector<long>{node}));
                }
                while (not unvisited.empty()) {
                    long currentNode = unvisited.top().first;
                    auto currentPathNodes = unvisited.top().second;
                    unvisited.pop();
                    if (not previousNodes.at(currentNode).empty()) {
                        for (long previousNode : previousNodes.at(currentNode)) {
                            auto newPath = currentPathNodes;
                            newPath.push_back(previousNode);
                            unvisited.push(make_pair(previousNode, newPath));
                        }
                    }
                    else
                        reversedPathNodes.emplace_back(currentPathNodes);
                }
                // For each path, select edges with highest weight
                for (auto &rpn : reversedPathNodes) {
                    vector<Edge> path;
                    bool isPathCompleted = true;
                    // Note the target node comes first
                    for (size_t i = rpn.size() - 1; i >= 1; i--) {
                        auto edges = kb.getEdgesBidirection(rpn[i], rpn[i-1]);
                        if (not edges.empty()) {
                            path.push_back(
                                    *max_element(edges.begin(), edges.end(),
                                                 [](const Edge &e1, const Edge &e2)
                                                 { return get<3>(e1) < get<3>(e2); }));
                        }
                        else {
                            isPathCompleted = false;
                            break;
                        }
                    }
                    if (isPathCompleted) {
                        float weight = accumulate(path.begin(), path.end(), 0,
                                                  [](float sum, const Edge &e) { return sum + get<3>(e); });
                        paths.push_back(make_pair(path, weight));
                    }
                }
            }
        }

        if (not paths.empty()) {
            // select path by path length first, then by best summed edge importance
            auto bestSubPath = max_element(paths.begin(), paths.end(),
                                         [](const pair<vector<Edge>, float> &p1,
                                            const pair<vector<Edge>, float> &p2){
                return p1.first.size() > p2.first.size() ||
                (p1.first.size() == p2.first.size() && p1.second < p2.second);
            })->first;
            bestPath.insert(bestPath.end(), bestSubPath.begin(), bestSubPath.end());

#ifdef DEBUG_DECISION
            cout << fmt::format("Level {}, Path found", level) << endl;
            for (size_t i = 0; i < paths.size(); i++) {
                cout << fmt::format("Path {}, weight={}: ", i, paths[i].second);
                for (auto &edge : paths[i].first)
                    cout << edgeToStringAnnotation(edge) << ", ";
                cout << endl;
            }
            cout << fmt::format("Level {} BestSubPath: ", level);
            for (auto &edge : bestSubPath)
                cout << edgeToStringAnnotation(edge) << ", ";
            cout << endl;
#endif
        }
        else {
#ifdef DEBUG_DECISION
            cout << fmt::format("Level {}, path not found, break", level) << endl;
#endif
            break;
        }
    }

    // Store start nodes of the first level
    get<2>(result).emplace_back(vector<long>(searchIds[0].begin(), searchIds[0].end()));

    long startNode = -1;
    if (not bestPath.empty()) {
        startNode = searchIds[0].find(get<0>(bestPath[0])) == searchIds[0].end() ?
                get<2>(bestPath[0]) : get<0>(bestPath[0]);
    }

    for (auto &edge : bestPath) {
        get<0>(result).emplace_back(edgeToAnnotation(edge));
        get<1>(result).emplace_back(edge);
        startNode = get<0>(edge) == startNode ? get<2>(edge) : get<0>(edge);
        get<2>(result).emplace_back(vector<long>{startNode});
    }
    return move(result);
}

KnowledgeMatcher::ChoiceResult
KnowledgeMatcher::findAvailableChoices(const vector<long> &visitedNodes,
                                       const vector<long> &previousNodes,
                                       bool pruneSimilarEdges) {
    ChoiceResult result;
    unordered_set<long> visitedSet(visitedNodes.begin(), visitedNodes.end());
    unordered_map<long, unordered_set<long>> addedEdges;
    if (previousNodes.size() == 1 and kb.isNodeComposite[previousNodes[0]])
        pruneSimilarEdges = false;
    for (long previousNode : previousNodes) {
        auto neighbors = kb.getNodeNeighbors(previousNode);
        for (long neighborNode : neighbors) {
            if (visitedSet.find(neighborNode) == visitedSet.end()) {
                auto edges = kb.getEdgesBidirection(previousNode, neighborNode);
                if (not edges.empty()) {
                    auto bestEdge = *max_element(edges.begin(), edges.end(), [](const Edge &e1, const Edge &e2) {
                        return get<3>(e1) < get<3>(e2);
                    });
                    if (not pruneSimilarEdges ||
                        addedEdges.find(get<1>(bestEdge)) == addedEdges.end() ||
                        addedEdges.at(get<1>(bestEdge)).find(get<2>(bestEdge))
                        == addedEdges.at(get<1>(bestEdge)).end()) {
                        get<0>(result).push_back(edgeToAnnotation(bestEdge));
                        get<1>(result).push_back(neighborNode);
                        get<2>(result).push_back(bestEdge);
                        addedEdges[get<1>(bestEdge)].insert(get<2>(bestEdge));
                    }
                }
            }
        }
    }
    return move(result);
}

KnowledgeMatcher::SourceAndTargetNodes
KnowledgeMatcher::matchSourceAndTargetNodes(const vector<int> &sourceSentence,
                                            const vector<int> &targetSentence,
                                            const vector<int> &sourceMask,
                                            const vector<int> &targetMask,
                                            size_t splitNodeMinimumEdgeNum,
                                            float splitNodeMinimumSimilarity) {
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    SourceAndTargetNodes result;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch,
                            splitNodeMinimumEdgeNum,
                            splitNodeMinimumSimilarity);
    for (auto &sm : sourceMatch)
        get<0>(result).push_back(kb.nodeMap.at(sm.second));
    for (auto &tm : targetMatch)
        get<1>(result).push_back(kb.nodeMap.at(tm.second));
    return move(result);
}

KnowledgeMatcher::MatchResult
KnowledgeMatcher::matchByNodeEmbedding(const vector<int> &sourceSentence,
                                       const vector<int> &targetSentence,
                                       const vector<int> &sourceMask,
                                       const vector<int> &targetMask,
                                       const vector<long> &disableNodes,
                                       int maxTimes, int maxDepth, int seed,
                                       int edgeTopK, int sourceContextRange, bool trim,
                                       size_t splitNodeMinimumEdgeNum,
                                       float splitNodeMinimumSimilarity,
                                       float stopSearchingEdgeIfSimilarityBelow,
                                       float sourceContextWeight) const {
    if (kb.nodeEmbeddingFileName.empty())
        throw invalid_argument("Knowledge base doesn't have an embedding file.");

#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: match by node embedding" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << fmt::format("disabledNodes: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "maxTimes: " << maxTimes << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "seed: " << seed << endl;
    cout << "edgeTopK: " << edgeTopK << endl;
    cout << "sourceContextRange: " << sourceContextRange << endl;
    cout << "trim: " << trim << endl;
    cout << "stopSearchingEdgeIfSimilarityBelow: " << stopSearchingEdgeIfSimilarityBelow << endl;
    cout << "================================================================================" << endl;
#endif
    // start token position of the node, tokens made up of the node
    unordered_map<size_t, vector<int>> sourceMatch, targetMatch;
    matchForSourceAndTarget(sourceSentence,
                            targetSentence,
                            sourceMask,
                            targetMask,
                            sourceMatch,
                            targetMatch,
                            splitNodeMinimumEdgeNum,
                            splitNodeMinimumSimilarity);

    if (sourceMatch.empty()) {
#ifdef DEBUG
        cout << "Source match result is empty, return" << endl;
#endif
        return move(MatchResult());
    }
    if (targetMatch.empty()) {
#ifdef DEBUG
        cout << "Target match result is empty, return" << endl;
#endif
        return move(MatchResult());
    }

    unordered_set<long> disabledNodeSet(disableNodes.begin(), disableNodes.end());

    // node ids in targetSentence (or sourceSentence if targetSentence is empty), their occurence times
    unordered_map<long, float> targetNodes;

    // random walk, transition possibility determined by similarity metric.
    MatchResult result;

    // node id, start pos, end pos
    vector<tuple<long, size_t, size_t>> nodes;
    vector<vector<long>> nodeContextNeighbors;

    unordered_map<pair<long, long>, float, PairHash> similarityCache;
    unordered_set<long> visitedEndNodes;

    // Convert source match result to source nodes and find contextual neighbors for each source node
    for (auto &sm : sourceMatch) {
        nodes.emplace_back(kb.nodeMap.at(sm.second), sm.first, sm.first + sm.second.size());
    }

    sort(nodes.begin(), nodes.end(), [](const tuple<long, size_t, size_t> &node1,
                                        const tuple<long, size_t, size_t> &node2) {
        return get<1>(node1) < get<1>(node2);
    });
    for (int i = 0; i < int(nodes.size()); i++) {
        vector<long> neighbors;
        for (int j = i - sourceContextRange; j <= i + sourceContextRange && sourceContextRange > 0; j++) {
            if (j >= 0 && j != i && j < int(nodes.size())) {
                neighbors.push_back(get<0>(nodes[j]));
            }
        }
        nodeContextNeighbors.emplace_back(neighbors);
    }

    for (auto &tm : targetMatch)
        targetNodes[kb.nodeMap.at(tm.second)] += 1;

    result.targetNodeNum = targetNodes.size();

    if (seed < 0) {
        random_device rd;
        seed = rd();
    }

    if (maxTimes < 2 * int(nodes.size()))
        cout << "Parameter maxTimes " << maxTimes << " is smaller than 2 * node size " << nodes.size()
             << ", which may result in insufficient exploration, consider increase maxTimes." << endl;

    float compositeNodesCountSum = 0;
    float corpusNodesCountSum = 0;
    for (auto &count : documentCountOfNodeInCorpus)
        corpusNodesCountSum += count.second;
    for (auto &count : kb.compositeComponentCount)
        compositeNodesCountSum += count.second;

    for (int i = 0; i < maxTimes; i++) {
        mt19937 gen(seed ^ i);

        // uniform sampling for efficient parallelization
        size_t nodeLocalIndex;
        long rootNode, currentNode;

        uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        nodeLocalIndex = nodeDist(gen);

        rootNode = currentNode = get<0>(nodes[nodeLocalIndex]);
        auto &neighbors = nodeContextNeighbors[nodeLocalIndex];

        VisitedPath path;
        path.round = i;
        path.root = rootNode;
        path.rootStartPos = get<1>(nodes[nodeLocalIndex]);
        path.rootEndPos = get<2>(nodes[nodeLocalIndex]);
        path.matchedFocusCount = 0;
        path.visitedNodes.insert(currentNode);

        unordered_map<long, float> oldTargetNodes = targetNodes;
        // Only works for first node
        for (auto neighbor : neighbors)
            targetNodes[neighbor] += sourceContextWeight;

#ifdef DEBUG_DECISION
        cout << fmt::format("Round {}", i) << endl;
        cout << "Compare target:" << endl;
        for (auto &tNode : targetNodes)
            cout << fmt::format("[{}:{}] {}", kb.nodes[tNode.first], tNode.first, tNode.second) << endl;
        cout << "================================================================================" << endl;
#endif
        for (int d = 0; d < maxDepth; d++) {
            vector<float> sim;
            vector<long> simTarget;
            bool hasOut = kb.edgeFromSource.find(currentNode) != kb.edgeFromSource.end();
            bool hasIn = kb.edgeToTarget.find(currentNode) != kb.edgeToTarget.end();
            size_t outSize = hasOut ? kb.edgeFromSource.at(currentNode).size() : 0;
            size_t inSize = hasIn ? kb.edgeToTarget.at(currentNode).size() : 0;
            sim.resize(outSize + inSize, 0);
            simTarget.resize(outSize + inSize, -1);

#ifdef DEBUG_DECISION
            cout << fmt::format("Current node: [{}:{}]", kb.nodes[currentNode], currentNode) << endl;
#endif
            for (size_t j = 0; j < outSize + inSize; j++) {
                size_t edgeIndex = j < outSize ?
                                   kb.edgeFromSource.at(currentNode)[j] :
                                   kb.edgeToTarget.at(currentNode)[j - outSize];
                auto &edge = kb.edges[edgeIndex];
                // disable disabled edges, reflexive edges & path to visted nodes
                long otherNodeId = j < outSize ? get<2>(edge) : get<0>(edge);

                if (kb.isEdgeDisabled[edgeIndex] ||
                    (path.visitedNodes.find(otherNodeId) != path.visitedNodes.end()) ||
                    (disabledNodeSet.find(otherNodeId) != disabledNodeSet.end())) {
                    sim[j] = 0;
#ifdef DEBUG_DECISION
                    cout << fmt::format("Skipping edge because: edge disabled [{}], node visited [{}], node disabled [{}]",
                                        kb.isEdgeDisabled[edgeIndex],
                                        path.visitedNodes.find(otherNodeId) != path.visitedNodes.end(),
                                        disabledNodeSet.find(otherNodeId) != disabledNodeSet.end()) << endl;
#endif
                } else {
                    float precision = 1e-4, recall = 1e-4,
                            precision_idf_sum = 0, recall_idf_sum = 0;
                    bool isNodeComposite = kb.isNodeComposite[otherNodeId];

                    if (isNodeComposite) {
                        for (auto &tNode : targetNodes) {
                            float subNodeSim = -1, simToEachTarget = -1;
                            long subNodeBest = -1;

                            for (auto &subNode : kb.compositeNodes.at(otherNodeId)) {
                                auto simPair = make_pair(tNode.first, subNode.first);
                                if (similarityCache.find(simPair) == similarityCache.end()) {
                                    simToEachTarget = kb.cosineSimilarity(subNode.first, tNode.first);
                                    similarityCache[simPair] = simToEachTarget;
                                } else {
                                    simToEachTarget = similarityCache.at(simPair);
                                }
                                if (simToEachTarget > subNodeSim) {
                                    subNodeSim = simToEachTarget;
                                    subNodeBest = subNode.first;
                                }
                            }

                            float tfidf = computeTfidf(tNode.first, corpusNodesCountSum,
                                                       targetNodes, documentCountOfNodeInCorpus);

#ifdef DEBUG_DECISION
                            cout << fmt::format("Target node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[subNodeBest], subNodeBest,
                                                subNodeSim, tfidf) << endl;
#endif
                            recall += subNodeSim * tfidf;
                            recall_idf_sum += tfidf;
                        }

                        for (auto &subNode : kb.compositeNodes.at(otherNodeId)) {
                            float tNodeSim = -1, simToEachTarget = -1;
                            long tNodeBest = -1;

                            for (auto &tNode : targetNodes) {
                                auto simPair = make_pair(tNode.first, subNode.first);
                                if (similarityCache.find(simPair) == similarityCache.end()) {
                                    simToEachTarget = kb.cosineSimilarity(subNode.first, tNode.first);
                                    similarityCache[simPair] = simToEachTarget;
                                } else {
                                    simToEachTarget = similarityCache.at(simPair);
                                }

                                if (simToEachTarget > tNodeSim) {
                                    tNodeSim = simToEachTarget;
                                    tNodeBest = tNode.first;
                                }
                            }

                            float tfidf = computeTfidf(subNode.first, compositeNodesCountSum,
                                                       kb.compositeNodes.at(otherNodeId),
                                                       kb.compositeComponentCount);
#ifdef DEBUG_DECISION
                            cout << fmt::format("Sub node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[subNode.first], subNode.first,
                                                kb.nodes[tNodeBest], tNodeBest,
                                                tNodeSim, tfidf) << endl;
#endif
                            precision += tNodeSim * tfidf;
                            precision_idf_sum += tfidf;
                        }
                        recall /= recall_idf_sum;
                        precision /= precision_idf_sum;
                    } else {
                        float tNodeSim = -1, simToEachTarget = -1;
                        long tNodeBest = -1;
                        for (auto &tNode : targetNodes) {
                            auto simPair = make_pair(tNode.first, otherNodeId);
                            if (similarityCache.find(simPair) == similarityCache.end()) {
                                simToEachTarget = kb.cosineSimilarity(otherNodeId, tNode.first);
                                similarityCache[simPair] = simToEachTarget;
                            } else {
                                simToEachTarget = similarityCache.at(simPair);
                            }

                            if (simToEachTarget > tNodeSim) {
                                tNodeSim = simToEachTarget;
                                tNodeBest = tNode.first;
                            }

                            float tfidf = computeTfidf(tNode.first, corpusNodesCountSum,
                                                       targetNodes, documentCountOfNodeInCorpus);

#ifdef DEBUG_DECISION
                            cout << fmt::format("Target node [{}:{}], node [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[otherNodeId], otherNodeId,
                                                simToEachTarget, tfidf) << endl;
#endif

                            recall += simToEachTarget * tfidf;
                            recall_idf_sum += tfidf;
                        }

#ifdef DEBUG_DECISION
                        cout << fmt::format("Node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                            kb.nodes[otherNodeId], otherNodeId,
                                            kb.nodes[tNodeBest], tNodeBest,
                                            tNodeSim, 1) << endl;
#endif
                        precision = tNodeSim;
                        recall = recall / recall_idf_sum;
                    }
                    recall = recall > 0 ? recall : 0;
                    precision = precision > 0 ? precision : 0;
#ifdef DEBUG_DECISION
                    cout << fmt::format("recall: {}, precision: {}", recall, precision) << endl;
#endif
                    float simTmp = (5 * recall * precision) / (recall + 4 * precision + 1e-6);
                    if (simTmp < stopSearchingEdgeIfSimilarityBelow)
                        simTmp = 0;
                    sim[j] = simTmp;
                }
#ifdef DEBUG_DECISION
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {} to target [{}:{}]",
                                    edgeIndex,
                                    kb.nodes[get<0>(edge)], get<0>(edge),
                                    kb.relationships[get<1>(edge)], get<1>(edge),
                                    kb.nodes[get<2>(edge)], get<2>(edge),
                                    edgeToStringAnnotation(edgeIndex),
                                    sim[j],
                                    simTarget[j] == -1 ? "" : kb.nodes[simTarget[j]], simTarget[j]) << endl;
#endif
            }

            if (d == 0)
                targetNodes.swap(oldTargetNodes);

            keepTopK(sim, edgeTopK);
            // break on meeting nodes with no match
            if (all_of(sim.begin(), sim.end(), [](float f) { return f <= 0; }))
                break;

            discrete_distribution<size_t> dist(sim.begin(), sim.end());
            size_t e = dist(gen);
            bool isEdgeOut = e < outSize;
            size_t selectedEdgeIndex = isEdgeOut ?
                                       kb.edgeFromSource.at(currentNode)[e] :
                                       kb.edgeToTarget.at(currentNode)[e - outSize];

            // move to next node
            currentNode = isEdgeOut ?
                          get<2>(kb.edges[selectedEdgeIndex]) :
                          get<0>(kb.edges[selectedEdgeIndex]);
            path.visitedNodes.insert(currentNode);
            path.edges.push_back(selectedEdgeIndex);
            path.similarities[selectedEdgeIndex] = sim[e];
            path.similarityTargets[selectedEdgeIndex] = simTarget[e];

#ifdef DEBUG_DECISION
            cout << endl;
            cout << "Choose edge " << selectedEdgeIndex << endl;
            cout << "Annotation: " << edgeToStringAnnotation(selectedEdgeIndex) << endl;
            cout << "--------------------------------------------------------------------------------" << endl;
#endif
        }
        if (trim)
            trimPath(path);
#ifdef DEBUG_DECISION
        cout << "================================================================================" << endl;
#endif
        result.visitedSubGraph.visitedPaths.push_back(path);
    }

    return move(result);
}

KnowledgeMatcher::MatchResult KnowledgeMatcher::joinMatchResults(const vector<MatchResult> &inMatchResults) const {
    // First assign new round number to each path in each match result,
    // to ensure that they are deterministically ordered, then join visited sub graphs
    // From:
    // [0, 1, 2, ..., 99], [0, 1, 2, ..., 99]
    // To:
    // [0, 1, 2, ..., 99], [100, 101, ..., 199]
    int offset = 0;
    vector<MatchResult> matchResults = inMatchResults;
    MatchResult joinedMatchResult;
    for (auto &result : matchResults) {
        for (auto &path : result.visitedSubGraph.visitedPaths)
            path.round += offset;
        offset += int(result.visitedSubGraph.visitedPaths.size());
        auto &vpIn = joinedMatchResult.visitedSubGraph.visitedPaths;
        auto &vpOut = result.visitedSubGraph.visitedPaths;
        vpIn.insert(vpIn.end(), vpOut.begin(), vpOut.end());
    }

    // Other properties of VisitedSubGraph are meant to be used by selectPaths and thus there is no need to join
    // Sort according round number to ensure deterministic ordering
    sort(joinedMatchResult.visitedSubGraph.visitedPaths.begin(),
         joinedMatchResult.visitedSubGraph.visitedPaths.end(),
         [](const VisitedPath &p1, const VisitedPath &p2) { return p1.round < p2.round; });

    return move(joinedMatchResult);
}

void KnowledgeMatcher::save(const string &archivePath) const {
    kb.save(archivePath);
}

void KnowledgeMatcher::load(const string &archivePath) {
    kb.load(archivePath);
}

template<typename T1, typename T2>
size_t KnowledgeMatcher::PairHash::operator()(const pair<T1, T2> &pair) const {
    return (hash<T1>()(pair.first) << 32) | hash<T2>()(pair.second);
}

vector<string> KnowledgeMatcher::matchResultPathsToStrings(const MatchResult &matchResult) const {
    vector<string> result;
    for (auto &path : matchResult.visitedSubGraph.visitedPaths) {
        vector<string> edgeStrings;
        for (size_t edgeIndex : path.edges) {
            edgeStrings.emplace_back(edgeToStringAnnotation(edgeIndex));
        }
        result.emplace_back(fmt::format("{}", fmt::join(edgeStrings.begin(), edgeStrings.end(), ",")));
    }
    return move(result);
}

vector<int> KnowledgeMatcher::edgeToAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    return move(edgeToAnnotation(edge));
}

string KnowledgeMatcher::edgeToStringAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    return move(edgeToStringAnnotation(edge));
}

vector<int> KnowledgeMatcher::edgeToAnnotation(const Edge &edge) const {
    vector<int> edgeAnno(kb.tokenizedNodes[get<0>(edge)]);
    auto &rel = kb.tokenizedRelationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    auto &tar = kb.tokenizedNodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

string KnowledgeMatcher::edgeToStringAnnotation(const Edge &edge) const {
    string edgeAnno(kb.nodes[get<0>(edge)]);
    edgeAnno += " ";
    auto &rel = kb.relationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    edgeAnno += " ";
    auto &tar = kb.nodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return move(edgeAnno);
}

float KnowledgeMatcher::computeTfidf(long node, float documentNodeCountSum,
                                     const unordered_map<long, float> &nodeCount,
                                     const unordered_map<long, float> &documentNodeCount) const {
    if (not isCorpusSet)
        return 1;
    float documentCount = 1;
    if (documentNodeCount.find(node) != documentNodeCount.end())
        documentCount += documentNodeCount.at(node);

    float idf = log(documentNodeCountSum / documentCount);

    float countSum = 0;
    for (auto &nc : nodeCount)
        countSum += nc.second;
    float tf = nodeCount.at(node) / countSum;
    return tf * idf;
}

void KnowledgeMatcher::matchForSourceAndTarget(const vector<int> &sourceSentence,
                                               const vector<int> &targetSentence,
                                               const vector<int> &sourceMask,
                                               const vector<int> &targetMask,
                                               unordered_map<size_t, vector<int>> &sourceMatch,
                                               unordered_map<size_t, vector<int>> &targetMatch,
                                               size_t splitNodeMinimumEdgeNum,
                                               float splitNodeMinimumSimilarity) const {
    if (not sourceMask.empty() && sourceMask.size() != sourceSentence.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for source sentence but size does not match, sentence: {}, mask: {}",
                sourceSentence.size(), sourceMask.size()));

    if (not targetMask.empty() && targetMask.size() != targetSentence.size())
        throw invalid_argument(fmt::format(
                "Mask is provided for target sentence but size does not match, sentence: {}, mask: {}",
                targetSentence.size(), targetMask.size()));

#ifdef DEBUG
    cout << "Begin node matching for source and target sentence" << endl;
#endif
    unordered_map<size_t, vector<vector<int>>> sourceMatches = kb.nodeTrie.matchForAll(sourceSentence, false);
    if (not sourceMask.empty()) {
        for (auto &matches : sourceMatches) {
            // Check if there exists any not masked position
            bool allMasked = true;
            for (size_t i = 0; i < matches.second.back().size(); i++) {
                if (sourceMask[matches.first + i] == 1) {
                    allMasked = false;
                }
            }
            if (allMasked) {
#ifdef DEBUG
                cout << fmt::format("Removing match [{}]@{} in source sentence",
                                    fmt::join(matches.second.back().begin(),
                                              matches.second.back().end(), ","), matches.first) << endl;
#endif
            } else
                normalizeMatch(sourceMatch, sourceMask, matches.first, matches.second.back(),
                               splitNodeMinimumEdgeNum, splitNodeMinimumSimilarity); // Insert longest
        }
    } else {
        for (auto &matches : sourceMatches)
            normalizeMatch(sourceMatch, sourceMask, matches.first, matches.second.back(), splitNodeMinimumEdgeNum,
                           splitNodeMinimumSimilarity); // Insert longest
    }
    if (targetSentence.empty())
        targetMatch = sourceMatch;
    else {
        unordered_map<size_t, vector<vector<int>>> targetMatches = kb.nodeTrie.matchForAll(targetSentence, false);
        if (not targetMask.empty()) {
            for (auto &matches : targetMatches) {
                // Check if there exists any not masked position
                bool allMasked = true;
                for (size_t i = 0; i < matches.second.back().size(); i++) {
                    if (targetMask[matches.first + i] == 1) {
                        allMasked = false;
                    }
                }
                if (allMasked) {
#ifdef DEBUG
                    cout << fmt::format("Removing match [{}]@{} in target sentence",
                                        fmt::join(matches.second.back().begin(),
                                                  matches.second.back().end(), ","), matches.first) << endl;
#endif
                } else
                    normalizeMatch(targetMatch, targetMask, matches.first, matches.second.back(),
                                   splitNodeMinimumEdgeNum, splitNodeMinimumSimilarity); // Insert longest
            }
        } else {
            for (auto &matches : targetMatches)
                normalizeMatch(targetMatch, targetMask, matches.first, matches.second.back(),
                               splitNodeMinimumEdgeNum, splitNodeMinimumSimilarity); // Insert longest
        }
    }
#ifdef DEBUG
    cout << "Finish node matching for source and target sentence" << endl;
#endif

}

void KnowledgeMatcher::normalizeMatch(unordered_map<size_t, vector<int>> &match,
                                      const vector<int> &mask,
                                      size_t position,
                                      const vector<int> &node,
                                      size_t splitNodeMinimumEdgeNum,
                                      float splitNodeMinimumSimilarity) const {
    long nodeId = kb.nodeMap.at(node);
    bool hasOut = kb.edgeFromSource.find(nodeId) != kb.edgeFromSource.end();
    bool hasIn = kb.edgeToTarget.find(nodeId) != kb.edgeToTarget.end();
    size_t outSize = hasOut ? kb.edgeFromSource.at(nodeId).size() : 0;
    size_t inSize = hasIn ? kb.edgeToTarget.at(nodeId).size() : 0;
    if (outSize + inSize < splitNodeMinimumEdgeNum) {
#ifdef DEBUG
        cout << fmt::format("Splitting node [{}:{}]", kb.nodes[nodeId], nodeId) << endl;
#endif
        unordered_map<size_t, vector<vector<int>>> subMatches = kb.nodeTrie.matchForAll(node, true);

        size_t currentOffset = 0;
        for (auto &subMatch : subMatches) {
            // When splitting node ABC
            // Might match [A, AB, ABC], [B, BC], [C] (each bracket is a subMatch)
            // For each subMatch X, if sim(X, ABC) > minimum similarity, insert it
            for (auto &subSubMatch : subMatch.second) {
                if (not isAllMasked(subMatch.first + position,
                                    subMatch.first + position + subSubMatch.size(),
                                    mask)) {
                    long subNodeId = kb.nodeMap.at(subSubMatch);
                    if (kb.cosineSimilarity(subNodeId, nodeId) > splitNodeMinimumSimilarity) {
                        match.emplace(position + subMatch.first, subSubMatch);
#ifdef DEBUG
                        cout << fmt::format("Splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
#endif
                    } else {
#ifdef DEBUG
                        cout << fmt::format("Ignore splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
#endif
                    }
                }
            }
        }
    } else {
        match.emplace(position, node);
    }
}

KnowledgeMatcher::SelectResult
KnowledgeMatcher::selectPaths(const KnowledgeMatcher::MatchResult &inMatchResult,
                              int maxEdges,
                              float discardEdgesIfRankBelow,
                              bool filterShortAccuratePaths) const {
#ifdef DEBUG
    cout << "Begin selecting paths" << endl;
#endif

    int remainingEdges = maxEdges;
    // uncovered similarity, length
    vector<pair<float, size_t>> pathRank;

    MatchResult matchResult = inMatchResult;
    auto &visitedSubGraph = matchResult.visitedSubGraph;

    if (filterShortAccuratePaths) {
        for (auto path = visitedSubGraph.visitedPaths.begin(); path != visitedSubGraph.visitedPaths.end();) {
            if (path->edges.size() >= 1 && path->similarities[path->edges[0]] > 0.5)
                path = visitedSubGraph.visitedPaths.erase(path);
            else
                path++;
        }
    }

    for (auto &path : visitedSubGraph.visitedPaths) {
        path.uncoveredEdges.insert(path.uncoveredEdges.end(), path.edges.begin(), path.edges.end());
        updatePath(path,
                   visitedSubGraph.coveredCompositeNodes,
                   visitedSubGraph.coveredNodePairs,
                   visitedSubGraph.sourceToTargetSimilarity,
                   remainingEdges);
        pathRank.emplace_back(make_pair(path.bestSimilarity, path.uncoveredEdges.size()));
    }

    while (remainingEdges > 0 && not pathRank.empty()) {
        size_t pathIndex = distance(pathRank.begin(),
                                    max_element(pathRank.begin(), pathRank.end(),
                                                [](const pair<float, size_t> &p1,
                                                   const pair<float, size_t> &p2) {
                                                    return p1.first < p2.first ||
                                                           p1.first == p2.first &&
                                                           p1.second > p2.second;
                                                }));
        if (pathRank[pathIndex].first <= discardEdgesIfRankBelow)
            break;
        auto &path = visitedSubGraph.visitedPaths[pathIndex];
        auto &addEdges = path.uncoveredEdges;
        auto &coveredEdges = visitedSubGraph.coveredSubGraph[make_pair(path.rootStartPos, path.rootEndPos)];
        coveredEdges.insert(coveredEdges.end(), addEdges.begin(), addEdges.end());
        for (size_t addEdgeIndex : addEdges) {
            long srcId = get<0>(kb.edges[addEdgeIndex]);
            long tarId = get<2>(kb.edges[addEdgeIndex]);
            bool isSrcComposite = kb.isNodeComposite[srcId];
            bool isTarComposite = kb.isNodeComposite[tarId];

            if (not isSrcComposite && not isTarComposite) {
                // Prevent inserting multiple edges connecting the same two nodes
                // If both of these two nodes are not composite
                visitedSubGraph.coveredNodePairs.insert(make_pair(srcId, tarId));
                visitedSubGraph.coveredNodePairs.insert(make_pair(tarId, srcId));
            } else {
                // Prevent inserting same composite nodes
                // (since composite nodes usually requires more space)
                if (isSrcComposite)
                    visitedSubGraph.coveredCompositeNodes.insert(srcId);
                if (isTarComposite)
                    visitedSubGraph.coveredCompositeNodes.insert(tarId);
            }
        }
        visitedSubGraph.sourceToTargetSimilarity[make_pair(path.root, path.bestSimilarityTarget)] = path.bestSimilarity;
        remainingEdges -= int(addEdges.size());

#ifdef DEBUG_DECISION
        cout << endl << "Rank result:" << endl;
        cout << "********************************************************************************" << endl;
        cout << "Root at position: " << path.rootStartPos << " Root: " << kb.nodes[path.root] << endl;
        cout << "Path rank: " << pathRank[pathIndex].first << " Length: " << pathRank[pathIndex].second << endl;
        cout << "Edges:" << endl;
        for (size_t addEdgeIndex : addEdges) {
            auto &addEdge = kb.edges[addEdgeIndex];
            if (path.similarityTargets[addEdgeIndex] != -1)
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {} to target [{}:{}]",
                                    addEdgeIndex,
                                    kb.nodes[get<0>(addEdge)], get<0>(addEdge),
                                    kb.relationships[get<1>(addEdge)], get<1>(addEdge),
                                    kb.nodes[get<2>(addEdge)], get<2>(addEdge),
                                    edgeToStringAnnotation(addEdgeIndex),
                                    path.similarities.at(addEdgeIndex),
                                    kb.nodes[path.similarityTargets[addEdgeIndex]],
                                    path.similarityTargets.at(addEdgeIndex)) << endl;
            else
                cout << fmt::format("{}: [{}:{}] --[{}:{}]--> [{}:{}], annotation [{}], "
                                    "similarity {}",
                                    addEdgeIndex,
                                    kb.nodes[get<0>(addEdge)], get<0>(addEdge),
                                    kb.relationships[get<1>(addEdge)], get<1>(addEdge),
                                    kb.nodes[get<2>(addEdge)], get<2>(addEdge),
                                    edgeToStringAnnotation(addEdgeIndex),
                                    path.similarities.at(addEdgeIndex)) << endl;
        }
        cout << "********************************************************************************" << endl;
#endif

        for (size_t i = 0; i < visitedSubGraph.visitedPaths.size(); i++) {
            auto &vPath = visitedSubGraph.visitedPaths[i];
            updatePath(vPath,
                       visitedSubGraph.coveredCompositeNodes,
                       visitedSubGraph.coveredNodePairs,
                       visitedSubGraph.sourceToTargetSimilarity,
                       remainingEdges);
            pathRank[i] = make_pair(vPath.bestSimilarity, vPath.uncoveredEdges.size());
        }
    }
    SelectResult result;
    for (auto &nodeSubGraph : visitedSubGraph.coveredSubGraph) {
        for (size_t edgeIndex : nodeSubGraph.second) {
            size_t startPos = nodeSubGraph.first.first, endPos = nodeSubGraph.first.second;
            get<0>(result[endPos]) = startPos;
            get<1>(result[endPos]).emplace_back(edgeToAnnotation(edgeIndex));
            get<2>(result[endPos]).emplace_back(get<3>(kb.edges[edgeIndex]));
        }
    }
#ifdef DEBUG
    cout << "Finish selecting paths" << endl;
#endif
    return move(result);
}

void KnowledgeMatcher::trimPath(VisitedPath &path) const {
    path.bestSimilarity = 0;
    path.bestSimilarityTarget = -1;
    size_t trimPosition = 0;
    for (size_t i = 0; i < path.edges.size(); i++) {
        float similarity = path.similarities.at(path.edges[i]);
        if (similarity >= path.bestSimilarity and similarity > 0) {
            path.bestSimilarity = similarity;
            path.bestSimilarityTarget = path.similarityTargets.at(path.edges[i]);
            trimPosition = i + 1;
        }
    }
#ifdef DEBUG_DECISION
    cout << "Trimming path to length " << trimPosition << endl;
#endif
    vector<size_t> newEdges(path.edges.begin(), path.edges.begin() + trimPosition);
    path.edges.swap(newEdges);
}

void KnowledgeMatcher::updatePath(VisitedPath &path,
                                  const unordered_set<long> &coveredCompositeNodes,
                                  const unordered_set<pair<long, long>, PairHash> &coveredNodePairs,
                                  const unordered_map<pair<long, long>, float, PairHash> &sourceToTargetSimilarity,
                                  int remainingEdges) const {
    const float similarityFocusMultiplier = 3;
    // Only effective for match by token, and doesn't affect bestSimilarityTarget when match by node/embedding
    float focusMultiplier = pow(similarityFocusMultiplier, float(path.matchedFocusCount));

    float bestSimilarity = 0;
    long bestSimilarityTarget = -1;
    vector<size_t> uncoveredEdges;
    // Only search edges from start that can fit into the remaining Edges
    for (size_t uEdge : path.uncoveredEdges) {
        if (remainingEdges <= 0)
            break;
        long srcId = get<0>(kb.edges[uEdge]);
        long tarId = get<2>(kb.edges[uEdge]);
        // Only add edges to uncovered list, if:
        // 1. Both of its ends are not composite nodes and both ends are not covered by some edge
        // 2. Any of its end is a composite node, and if it is, the composite node must be not covered
        if ((not kb.isNodeComposite[srcId] &&
             not kb.isNodeComposite[tarId] &&
             coveredNodePairs.find(make_pair(srcId, tarId)) == coveredNodePairs.end() &&
             coveredNodePairs.find(make_pair(tarId, srcId)) == coveredNodePairs.end()) ||
            ((kb.isNodeComposite[srcId] || kb.isNodeComposite[tarId]) &&
             coveredCompositeNodes.find(srcId) == coveredCompositeNodes.end() &&
             coveredCompositeNodes.find(tarId) == coveredCompositeNodes.end())) {
            uncoveredEdges.emplace_back(uEdge);
            if (path.similarities.at(uEdge) * focusMultiplier > bestSimilarity) {
                auto pair = make_pair(path.root, path.similarityTargets.at(uEdge));
                // Consider a path from some root node R to some target node T
                // If its similarity is larger than another path from same root
                // R to target node Tin the covered set
                // Then make this path the best path from R -> T

                // In the mean time, since a path can be similar to multiple nodes
                // (or -1 indicating that do not compare to a single node but a group of nodes)
                // select the most similar target node T* amoung all target nodes {T}
                bool isBetterThanCurrent = (
                        sourceToTargetSimilarity.find(pair) == sourceToTargetSimilarity.end() ||
                        sourceToTargetSimilarity.at(pair) < path.similarities.at(uEdge) * focusMultiplier);
                if (isBetterThanCurrent or pair.second == -1) {
                    bestSimilarity = path.similarities.at(uEdge) * focusMultiplier;
                    bestSimilarityTarget = path.similarityTargets.at(uEdge);
                }
            }
            remainingEdges--;
        }
    }
    path.bestSimilarity = bestSimilarity;
    path.bestSimilarityTarget = bestSimilarityTarget;
    path.uncoveredEdges.swap(uncoveredEdges);
}

void KnowledgeMatcher::keepTopK(vector<float> &weights, int k) {
    if (k < 0)
        return;
    size_t size = weights.size();
    k = min(k, int(size));
    if (k == int(size))
        return;
    auto result = xt::argpartition(xt::adapt(weights, vector<size_t>{weights.size()}), size_t(size - k));
    for (size_t i = 0; i < size - k; i++) {
        weights[result[i]] = 0;
    }
}
