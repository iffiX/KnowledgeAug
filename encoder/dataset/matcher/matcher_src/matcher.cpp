#define FMT_HEADER_ONLY

#include "matcher.h"
#include "tqdm.h"
#include "fmt/format.h"
#include "xtensor/xio.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xtensor.hpp"
#include "highfive/H5File.hpp"
#include <regex>
#include <stack>
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
//#ifdef DEBUG
//    cout << "Matched results:" << endl;
//    for (auto &r : result)
//        cout << fmt::format("[{}]", fmt::join(r.begin(), r.end(), ",")) << endl;
//#endif
    if (not allowSubMatch and not result.empty())
        result.erase(result.begin(), result.begin() + result.size() - 1);
    return std::move(result);
}

unordered_map<size_t, vector<vector<int>>> Trie::matchForAll(const vector<int> &sentence, bool allowSubMatch) const {
    unordered_map<size_t, vector<vector<int>>> result;
    for (size_t i = 0; i < sentence.size();) {
        vector<vector<int>> matches = std::move(matchForStart(sentence, i, allowSubMatch));
        if (not matches.empty()) {
            size_t match_size = matches.front().size();
            result.emplace(i, matches);
            i += match_size;
        } else
            i++;
    }
    return std::move(result);
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
    return std::move(ids);
}

vector<Edge>
KnowledgeBase::findEdges(const vector<string> &source_nodes,
                         const vector<string> &relations,
                         const vector<string> &target_nodes,
                         bool quiet) const {
    if (source_nodes.size() != target_nodes.size() or source_nodes.size() != relations.size())
        throw invalid_argument("Source node size, relation size, target node size should be the same");
    vector<Edge> result;
    unordered_map<string, long> relationship_map;
    long rel = 0;
    for (auto & relationship : relationships)
        relationship_map[relationship] = rel++;

    auto source_node_ids = findNodes(source_nodes, quiet);
    auto target_node_ids = findNodes(target_nodes, quiet);
    vector<long> relation_ids;
    for (auto & relation : relations) {
        if (relationship_map.find(relation) != relationship_map.end()) {
            relation_ids.push_back(relationship_map.at(relation));
        }
        else {
            if (not quiet)
                throw invalid_argument(fmt::format("Relation {} not found", relation));
            else
                relation_ids.push_back(-1);
        }
    }

    for (size_t i = 0; i< source_nodes.size(); i++) {
        auto edges = getEdges(source_node_ids[i], target_node_ids[i]);
        bool found = false;
        for (auto & edge : edges) {
            if (get<1>(edge) == relation_ids[i]) {
                found = true;
                result.emplace_back(edge);
                break;
            }
        }
        if (not found) {
            if (not quiet)
                throw invalid_argument(fmt::format("Edge ({}) -[{}]-> ({}) not found", source_nodes[i], relations[i],
                                                   target_nodes[i]));
            else
                result.emplace_back(make_tuple(-1, -1, -1, 0, ""));
        }
    }
    return std::move(result);
}

unordered_set<long> KnowledgeBase::getNodeNeighbors(long node) const {
    if (adjacency.find(node) == adjacency.end())
        throw invalid_argument("Node doesn't exist");
    return std::move(adjacency.at(node));
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

    return std::move(result);
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
    return std::move(result);
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

tuple<size_t, size_t> KnowledgeBase::getInAndOutEdgeNumOfNode(long node, bool excludeCompositeNodes) const {
    size_t inSize = 0, outSize = 0;
    // Also work for composite nodes since they only have singe direction
    // connection with non-composite component nodes.

    if (edgeToTarget.find(node) != edgeToTarget.end()) {
        for (size_t edgeIndex : edgeToTarget.at(node)) {
            if (not isNodeComposite[get<0>(edges[edgeIndex])] or not excludeCompositeNodes)
                inSize += 1;
        }
    }
    if (edgeFromSource.find(node) != edgeFromSource.end()) {
        for (size_t edgeIndex : edgeFromSource.at(node)) {
            if (not isNodeComposite[get<2>(edges[edgeIndex])] or not excludeCompositeNodes)
            outSize += 1;
        }
    }
    return make_tuple(inSize, outSize);
}

long KnowledgeBase::getCompositeStart() const {
    for (long i = 0; i < isNodeComposite.size(); i++) {
        if (isNodeComposite[i])
            return i;
    }
    return isNodeComposite.size();
}

long KnowledgeBase::addCompositeNode(const string &compositeNode,
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
        auto sizes = getInAndOutEdgeNumOfNode(subNodeId, true);

        if (get<0>(sizes) + get<1>(sizes) < splitNodeMinimumEdgeNum) {
//#ifdef DEBUG
//            cout << fmt::format("Splitting node [{}:{}]", nodes[subNodeId], subNodeId) << endl;
//#endif
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

//#ifdef DEBUG
//                            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
//                                                nodes[baseNodeId], baseNodeId,
//                                                nodes[newNodeId], newNodeId) << endl;
//#endif
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

//#ifdef DEBUG
//            cout << fmt::format("Adding component [{}:{}] to composite node [{}:{}]",
//                                nodes[subNodeId], subNodeId,
//                                nodes[newNodeId], newNodeId) << endl;
//#endif
        }
    }
    compositeNodes.emplace(newNodeId, components);
    nodeToIndex[compositeNode] = newNodeId;
    return newNodeId;
}

size_t KnowledgeBase::addCompositeEdge(long sourceNodeId, long relationId, long compositeNodeId) {
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
//#ifdef DEBUG
//    cout << fmt::format("Connecting node [{}:{}] to composite node [{}:{}] with relation [{}:{}]",
//                        nodes[sourceNodeId], sourceNodeId,
//                        nodes[compositeNodeId], compositeNodeId,
//                        relationships[relationId], relationId) << endl;
//#endif
    return edgeIndex;
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
    return std::move(cista::raw::vector<T>(vec.begin(), vec.end()));
}

template<typename T>
vector<T> KnowledgeBase::archiveVector2Vector(const cista::raw::vector<T> &vec) {
    return std::move(vector<T>(vec.begin(), vec.end()));
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

float KnowledgeMatcher::computeFBetaScore(long node, const vector<long> &targetNodes,
                                          float beta) const {
    unordered_map<pair<long, long>, float, PairHash> similarityCache;
    unordered_map<long, float> targetNodeCounts;
    for (long targetNode : targetNodes)
        targetNodeCounts[targetNode] += 1;
    return computeFBetaScoreWithCache(node, targetNodeCounts, similarityCache, beta);
}

string KnowledgeMatcher::findClosestConcept(string targetConcept, const vector<string> &concepts) const {
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
                                   bool findTarget,
                                   size_t maxDepthForEachNode,
                                   size_t minLevelsBeforeCheckingTargetReached,
                                   size_t splitNodeMinimumEdgeNum,
                                   float splitNodeMinimumSimilarity) const {
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
    cout << "minLevelsBeforeCheckingTargetReached: " << minLevelsBeforeCheckingTargetReached << endl;
    cout << "splitNodeMinimumEdgeNum: " << splitNodeMinimumEdgeNum << endl;
    cout << "splitNodeMinimumSimilarity: " << splitNodeMinimumSimilarity << endl;
    cout << "================================================================================" << endl;
#endif
    // start token position of the node, tokens made up of the node
    vector<pair<size_t, vector<int>>> sourceMatch, targetMatch;
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
        return std::move(PathResult());
    }

    PathResult result;

    // node ids in sourceSentence / targetSentence, their occurence times
    unordered_map<long, float> sourceNodeOccurrences;
    unordered_map<long, float> targetNodeOccurrences;

    for (auto &sm : sourceMatch)
        sourceNodeOccurrences[kb.nodeMap.at(sm.second)] += 1;

    for (auto &tm : targetMatch)
        targetNodeOccurrences[kb.nodeMap.at(tm.second)] += 1;

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
    if (findTarget and not targetNodes.empty())
        searchIds.emplace_back(unordered_set<long>(targetNodes.begin(), targetNodes.end()));

#ifdef DEBUG_DECISION
    cout << endl << "SearchIds:" << endl;
    for (size_t i = 0; i < searchIds.size(); i++) {
        cout << fmt::format("Level {}: ", i);
        for (long node : searchIds[i])
            cout << fmt::format("({}) {}, ", node, kb.nodes[node]);
        cout << endl;
    }
#endif

    vector<vector<Edge>> bestPath;
    for (size_t level = 0; level < searchIds.size() - 1; level++)
    {
        vector<pair<vector<Edge>, float>> paths;

        size_t bestLevelDepth = maxDepthForEachNode;

        // rank start nodes by their similarity to the next level nodes
        vector<long> startNodes;
        vector<pair<long, float>> rank;
        for (long node : searchIds[level]) {
            if (kb.isNodeComposite[node]) {
                for (auto &component : kb.compositeNodes.at(node))
                    startNodes.push_back(component.first);
            }
            else {
                startNodes.push_back(node);
            }
        }
        for (long startNode : startNodes) {
            float bestSimilarity = -1;
            for(long nextNode : searchIds[level + 1]) {
                if (not kb.isNodeComposite[nextNode])
                    bestSimilarity= kb.cosineSimilarity(startNode, nextNode);
                else {
                    for (auto component : kb.compositeNodes.at(nextNode)) {
                        float similarity = kb.cosineSimilarity(startNode, component.first);
                        bestSimilarity = similarity > bestSimilarity ? similarity : bestSimilarity;
                    }
                }
            }
            rank.push_back(make_pair(startNode, bestSimilarity));
        }
        sort(rank.begin(), rank.end(), [](const pair<long, float> &a, const pair<long, float> &b){
            return a.second > b.second;
        });
        startNodes.clear();
        for (auto &node : rank)
            startNodes.push_back(node.first);

        bool reached = false;
        if (level >= minLevelsBeforeCheckingTargetReached) {
            for (long startNode: startNodes) {
                if (searchIds[level + 1].find(startNode) != searchIds[level + 1].end()) {
                    reached = true;
                    break;
                }
            }
        }


        // Note: currently, if several levels share the same start nodes, they are not deduplicated
#ifdef DEBUG_DECISION
        cout << endl << fmt::format("Level {} Start Nodes:", level) << endl;
        for (long node : startNodes)
            cout << fmt::format("({}) {}, ", node, kb.nodes[node]);
        cout << endl;
        if (reached)
            cout << "Target node reached" << endl;
#endif
        if (reached)
            break;

        for (long startNode : startNodes) {
            vector<bool> visited(kb.nodes.size(), false);
            // Prevent cycles
            for (auto &subPath : bestPath) {
                for (auto &edge : subPath) {
                    visited[get<0>(edge)] = true;
                    visited[get<2>(edge)] = true;
                }
            }
            unordered_set<long> lastStepVisitedNodes;
            unordered_map<long, vector<long>> previousNodes;
            visited[startNode] = true;
            lastStepVisitedNodes.insert(startNode);
            previousNodes[startNode] = {};

            bool found = false;
            size_t step;

#ifdef DEBUG_DECISION
            cout << fmt::format("Current start node: ({}) {}", startNode, kb.nodes[startNode]) << endl;
#endif
            for (step = 0; step < bestLevelDepth; step++) {
                unordered_set<long> newVisitedNodes;
                for (long lastNode : lastStepVisitedNodes) {
                    // Forbid exploration after composite nodes
                    if (not kb.isNodeComposite[lastNode]) {
                        auto neighbors = kb.getNodeNeighbors(lastNode);
                        for (long neighbor : neighbors) {
                            // If the neighbor node is only visited in the current step
                            if (not visited[neighbor]) {
                                previousNodes[neighbor].push_back(lastNode);
                                newVisitedNodes.insert(neighbor);
                            }
                        }
                    }
                }
#ifdef DEBUG_DECISION
                cout << fmt::format("Step {} visited nodes {}", step, newVisitedNodes.size()) << endl;
#endif
                for (long newVisitedNode : newVisitedNodes)
                    visited[newVisitedNode] = true;
                found = unordered_set_has_intersection(searchIds[level + 1].begin(),
                                                       searchIds[level + 1].end(),
                                                       newVisitedNodes.begin(),
                                                       newVisitedNodes.end());
                lastStepVisitedNodes.swap(newVisitedNodes);
                if (found)
                    break;
            }
            if (found) {
                // construct paths in the reverse order
                bool isAnyPathAvailable = false;
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
                    else if (currentPathNodes.size() > 1)
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
                        isAnyPathAvailable = true;
                        float weight = accumulate(path.begin(), path.end(), 0,
                                                  [](float sum, const Edge &e) { return sum + get<3>(e); });
                        paths.push_back(make_pair(path, weight));
                    }
                    else {
#ifdef DEBUG_DECISION
                        cout << fmt::format("Reject potential path: ");
                        // size_t may cause underflow
                        for (long i = rpn.size() - 1; i >= 0; i--)
                            cout << fmt::format("({}) {}, ", rpn[i], kb.nodes[rpn[i]]);
                        cout << endl;
#endif
                    }
                }

                if (isAnyPathAvailable and bestLevelDepth > step + 1) {
                    // Update max allowed depth
                    bestLevelDepth = step + 1;
#ifdef DEBUG_DECISION
                    cout << fmt::format("Update bestLevelDepth to {}", bestLevelDepth) << endl;
#endif
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
            bestPath.emplace_back(bestSubPath);

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

    for (size_t i = 0; i < bestPath.size(); i++) {
        auto &bestSubPath = bestPath[i];

        // Store annotations of each edge
        get<0>(result).emplace_back(vector<vector<int>>{});
        auto &edgeAnnotations = get<0>(result).back();
        for(auto &edge : bestSubPath)
            edgeAnnotations.push_back(edgeToAnnotation(edge));

        // Store path
        get<1>(result).emplace_back(bestSubPath);

        // Store start nodes (of next level)
        if (i < intermediateNodes.size()) {
            long startNode = *searchIds[i + 1].begin();
            if (kb.isNodeComposite[startNode]) {
                vector<long> components;
                for (auto &component : kb.compositeNodes.at(startNode))
                    components.push_back(component.first);
                get<2>(result).emplace_back(components);
            }
            else
                get<2>(result).emplace_back(vector<long>{startNode});
        }
        else {
            // This sub path leads to the target nodes
            auto &lastEdge = bestSubPath.back();
            if (searchIds.back().find(get<0>(lastEdge)) != searchIds.back().end())
                get<2>(result).emplace_back(vector<long>{get<0>(lastEdge)});
            else
                get<2>(result).emplace_back(vector<long>{get<2>(lastEdge)});
        }
    }
    return std::move(result);
}

KnowledgeMatcher::ChoiceResult
KnowledgeMatcher::findAvailableChoices(const vector<long> &visitedNodes,
                                       const vector<long> &startNodes,
                                       const vector<long> &targetNodes,
                                       const vector<long> &allowedCompositeNodes,
                                       int maxDepth,
                                       bool parallel,
                                       bool findTarget,
                                       bool findComposite,
                                       bool filterCompositeNodesByFBeta,
                                       float minimumFBeta) const {
#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: findAvailableChoices" << endl;
    cout << fmt::format("visitedNodes: [{}]",
                        fmt::join(visitedNodes.begin(), visitedNodes.end(), ",")) << endl;
    cout << fmt::format("startNodes: [{}]",
                        fmt::join(startNodes.begin(), startNodes.end(), ",")) << endl;
    cout << fmt::format("targetNodes: [{}]",
                        fmt::join(targetNodes.begin(), targetNodes.end(), ",")) << endl;
    cout << "maxDepth: " << maxDepth << endl;
    cout << "================================================================================" << endl;
#endif

    ChoiceResult result;
    unordered_set<long> targetNodesSet(targetNodes.begin(), targetNodes.end());
    vector<bool> allowedCompositeNodesIndex(kb.nodes.size(), false);
    unordered_map<pair<long, long>, float, PairHash> similarityCache;
    unordered_map<long, float> context;

    for (long startNode : startNodes)
        context[startNode] += 1;

    for (long allowedCompositeNode : allowedCompositeNodes)
        allowedCompositeNodesIndex[allowedCompositeNode] = true;

#pragma omp parallel for num_threads(4) default(none) private(similarityCache) \
    shared(cout, visitedNodes, startNodes, targetNodes, allowedCompositeNodes, \
           maxDepth, findTarget, findComposite, filterCompositeNodesByFBeta, minimumFBeta,\
           targetNodesSet, allowedCompositeNodesIndex, context, result) \
    if (parallel)
    for (long startNode : startNodes) {
        unordered_set<long> lastStepVisitedNodes;
        unordered_map<long, vector<long>> previousNodes;
        vector<bool> visited(kb.nodes.size(), false);

        for (long visitedNode : visitedNodes)
            visited[visitedNode] = true;
        visited[startNode] = true;
        lastStepVisitedNodes.insert(startNode);
        previousNodes[startNode] = {};

        size_t step;
        for (step = 0; step < maxDepth; step++) {
            unordered_set<long> newVisitedNodes;
            for (long lastNode : lastStepVisitedNodes) {
                // We only want paths made up of hops between non composite nodes to either:
                // 1) one of a target node
                // 2) one of a composite node
                if (not kb.isNodeComposite[lastNode]) {
                    auto neighbors = kb.getNodeNeighbors(lastNode);
                    for (long neighbor : neighbors) {
                        // If the neighbor node is only visited in the current step
                        if (not visited[neighbor] and
                            (not kb.isNodeComposite[neighbor]
                             or allowedCompositeNodes.empty()
                             or allowedCompositeNodesIndex[neighbor])) {
                            previousNodes[neighbor].push_back(lastNode);
                            newVisitedNodes.insert(neighbor);
                        }
                    }
                }
            }
#ifdef DEBUG_DECISION
            cout << fmt::format("Step {} visited nodes {}", step, newVisitedNodes.size()) << endl;
#endif
            for (long newVisitedNode : newVisitedNodes)
                visited[newVisitedNode] = true;
#ifdef DEBUG_DECISION
            bool found = (not targetNodes.empty() and
                          unordered_set_has_intersection(targetNodesSet.begin(),
                                                         targetNodesSet.end(),
                                                         newVisitedNodes.begin(),
                                                         newVisitedNodes.end()));
            if (found)
                cout << fmt::format("Step {} found target nodes", step) << endl;
#endif
            lastStepVisitedNodes.swap(newVisitedNodes);
        }

        // construct paths in the reverse order
        stack<pair<long, vector<long>>> unvisited;
        vector<vector<long>> reversedPathNodes;

        // Add visited target nodes as reversed starts
        if (findTarget) {
            for (long node : targetNodes) {
                if (previousNodes.find(node) != previousNodes.end())
                    unvisited.push(make_pair(node, vector<long>{node}));
            }
        }
        // Find all visited composite nodes and add them as reversed starts
        // Note that visited nodes may not be visited in the current traversal (see argument visitedNodes)
        // exclude them
        if (findComposite) {
            for (long node = 0; node < kb.nodes.size(); node++) {
                if (visited[node] and kb.isNodeComposite[node]
                    and previousNodes.find(node) != previousNodes.end()
                    and targetNodesSet.find(node) == targetNodesSet.end()
                    and (allowedCompositeNodes.empty() or allowedCompositeNodesIndex[node])
                    and (not filterCompositeNodesByFBeta
                         or computeFBetaScoreWithCache(node, context, similarityCache, 2) > minimumFBeta))
                    unvisited.push(make_pair(node, vector<long>{node}));
            }
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
            else if (currentPathNodes.size() > 1)
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
                // Store annotations of each edge
                auto edgeAnnotations = vector<vector<int>>{};
                for(auto &edge : path)
                    edgeAnnotations.push_back(edgeToAnnotation(edge));

                // Store end nodes, for composite end nodes, store components
                vector<long> endNodes;
                if (kb.isNodeComposite[rpn[0]]) {
                    for (auto &component : kb.compositeNodes.at(rpn[0]))
                        endNodes.push_back(component.first);
                }
                else
                    endNodes.push_back(rpn[0]);

#pragma omp critical
                {
                    get<0>(result).emplace_back(edgeAnnotations);
                    get<1>(result).emplace_back(endNodes);
                    get<2>(result).emplace_back(rpn[0]);
                    get<3>(result).emplace_back(path);
                };
            }
        }
    }
#ifdef DEBUG_DECISION
    cout << fmt::format("Start nodes: [");
    for (long node : startNodes)
        cout << kb.nodes[node] << ", ";
    cout << "]" << endl;
    cout << fmt::format("Target nodes: [");
    for (long node : targetNodes)
        cout << kb.nodes[node] << ", ";
    cout << "]" << endl;
    cout << fmt::format("Paths:") << endl;
    for (size_t i = 0; i < get<1>(result).size(); i++) {
        cout << fmt::format("Path {}: ", i);
        for (auto &edge : get<1>(result)[i])
            cout << edgeToStringAnnotation(edge) << ", ";
        cout << endl;
    }
#endif
    return std::move(result);
}

vector<vector<vector<int>>>
KnowledgeMatcher::subPathsToAnnotations(const vector<vector<Edge>> &subPaths) const {
    vector<vector<vector<int>>> result;
    for (auto &subPath : subPaths) {
        result.emplace_back(vector<vector<int>>{});
        auto &subPathAnnotations = result.back();
        for (auto &edge : subPath) {
            subPathAnnotations.emplace_back(edgeToAnnotation(edge));
        }
    }
    return std::move(result);
}

vector<vector<string>>
KnowledgeMatcher::subPathsToStringAnnotations(const vector<vector<Edge>> &subPaths,
                                        const vector<string> &relationshipTemplates,
                                        bool prioritizeOriginalAnnotation,
                                        bool lowerCase) const {
    if (relationshipTemplates.size() != kb.relationships.size())
        throw invalid_argument(fmt::format(
                "Relationship templates size {} doesn't match with relationship size {}",
                relationshipTemplates.size(), kb.relationships.size()
        ));
    vector<vector<string>> result;
    for (auto &subPath : subPaths) {
        result.emplace_back(vector<string>{});
        auto &subPathStringAnnotations = result.back();
        for (auto &edge : subPath) {
            if (prioritizeOriginalAnnotation and get<4>(edge).length() > 0) {
                string annotation = get<4>(edge);
                if (lowerCase)
                    for_each(annotation.begin(), annotation.end(), [](char &c){ c = tolower(c); });
                // remove leading, trailing and extra white spaces
                subPathStringAnnotations.emplace_back(regex_replace(annotation, regex("^ +| +$|( ) +"), "$1"));
            }
            else {
                string annotation = fmt::format(relationshipTemplates[get<1>(edge)],
                                                kb.nodes[get<0>(edge)],
                                                kb.nodes[get<2>(edge)]);
                if (lowerCase)
                    for_each(annotation.begin(), annotation.end(), [](char &c){ c = tolower(c); });
                subPathStringAnnotations.emplace_back(annotation);
            }
        }
    }
    return std::move(result);
}

KnowledgeMatcher::SourceAndTargetNodes
KnowledgeMatcher::matchSourceAndTargetNodes(const vector<int> &sourceSentence,
                                            const vector<int> &targetSentence,
                                            const vector<int> &sourceMask,
                                            const vector<int> &targetMask,
                                            size_t splitNodeMinimumEdgeNum,
                                            float splitNodeMinimumSimilarity) const {
#ifdef DEBUG
    cout << "================================================================================" << endl;
    cout << "Method: matchSourceAndTargetNodes" << endl;
    cout << fmt::format("sourceSentence: [{}]",
                        fmt::join(sourceSentence.begin(), sourceSentence.end(), ",")) << endl;
    cout << fmt::format("targetSentence: [{}]",
                        fmt::join(targetSentence.begin(), targetSentence.end(), ",")) << endl;
    cout << fmt::format("sourceMask: [{}]",
                        fmt::join(sourceMask.begin(), sourceMask.end(), ",")) << endl;
    cout << fmt::format("targetMask: [{}]",
                        fmt::join(targetMask.begin(), targetMask.end(), ",")) << endl;
    cout << "splitNodeMinimumEdgeNum: " << splitNodeMinimumEdgeNum << endl;
    cout << "splitNodeMinimumSimilarity: " << splitNodeMinimumSimilarity << endl;
    cout << "================================================================================" << endl;
#endif
    vector<pair<size_t, vector<int>>> sourceMatch, targetMatch;
    unordered_set<long> sourceNodesSet, targetNodesSet;
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
        sourceNodesSet.insert(kb.nodeMap.at(sm.second));
    for (auto &tm : targetMatch)
        targetNodesSet.insert(kb.nodeMap.at(tm.second));
    for (auto &sn : sourceNodesSet)
        get<0>(result).push_back(sn);
    for (auto &tn : targetNodesSet)
        get<1>(result).push_back(tn);
    return std::move(result);
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

vector<int> KnowledgeMatcher::edgeToAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    return std::move(edgeToAnnotation(edge));
}

string KnowledgeMatcher::edgeToStringAnnotation(size_t edgeIndex) const {
    const Edge &edge = kb.edges[edgeIndex];
    return std::move(edgeToStringAnnotation(edge));
}

vector<int> KnowledgeMatcher::edgeToAnnotation(const Edge &edge) const {
    vector<int> edgeAnno(kb.tokenizedNodes[get<0>(edge)]);
    auto &rel = kb.tokenizedRelationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    auto &tar = kb.tokenizedNodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return std::move(edgeAnno);
}

string KnowledgeMatcher::edgeToStringAnnotation(const Edge &edge) const {
    string edgeAnno(kb.nodes[get<0>(edge)]);
    edgeAnno += " ";
    auto &rel = kb.relationships[get<1>(edge)];
    edgeAnno.insert(edgeAnno.end(), rel.begin(), rel.end());
    edgeAnno += " ";
    auto &tar = kb.nodes[get<2>(edge)];
    edgeAnno.insert(edgeAnno.end(), tar.begin(), tar.end());
    return std::move(edgeAnno);
}

size_t KnowledgeMatcher::componentIntersection(const unordered_map<long, float> &sourceNodes,
                                               const unordered_map<long, float> &targetNodes) {
    size_t count = 0;
    for (auto &node : sourceNodes)
        if (targetNodes.find(node.first) != targetNodes.end())
            count += 1;
    return count;
}

float KnowledgeMatcher::computeFBetaScoreWithCache(long node, const unordered_map<long, float> &targetNodes,
                                                   unordered_map<pair<long, long>, float, PairHash> &similarityCache,
                                                   float beta) const {
    float precision = 1e-4, recall = 1e-4, precision_idf_sum = 0, recall_idf_sum = 0;
    bool isNodeComposite = kb.isNodeComposite[node];

    if (isNodeComposite) {
        for (auto &tNode : targetNodes) {
            float subNodeSim = -1, simToEachTarget = -1;
            long subNodeBest = -1;

            for (auto &subNode : kb.compositeNodes.at(node)) {
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

            // For target nodes, the document is themselves (the target sentence)
            // the corpus is the preset corpus
            float tfidf = isCorpusSet ?
                          computeTfidf(tNode.first,
                                       computeNodeCountSum(targetNodes),
                                       float(corpusSize),
                                       targetNodes,
                                       documentCountOfNodeInCorpus) : 1;

#ifdef DEBUG_DECISION
            cout << fmt::format("Target node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[subNodeBest], subNodeBest,
                                                subNodeSim, tfidf) << endl;
#endif
            recall += subNodeSim * tfidf;
            recall_idf_sum += tfidf;
        }

        for (auto &subNode : kb.compositeNodes.at(node)) {
            float tNodeSim = -1, simToEachTarget = -1;
            long tNodeBest = -1;

            for (auto &tNode : targetNodes) {
                auto simPair = make_pair(tNode.first, subNode.first);
                simToEachTarget = similarityCache.at(simPair);

                if (simToEachTarget > tNodeSim) {
                    tNodeSim = simToEachTarget;
                    tNodeBest = tNode.first;
                }
            }

            // For composite nodes, the document is themselves (the composite node sentence)
            // the corpus is the set of all composite nodes
            float tfidf = isCorpusSet ?
                          computeTfidf(subNode.first,
                                       computeNodeCountSum(kb.compositeNodes.at(node)),
                                       float(kb.compositeNodes.size()),
                                       kb.compositeNodes.at(node),
                                       kb.compositeComponentCount) : 1;
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
            auto simPair = make_pair(tNode.first, node);
            if (similarityCache.find(simPair) == similarityCache.end()) {
                simToEachTarget = kb.cosineSimilarity(node, tNode.first);
                similarityCache[simPair] = simToEachTarget;
            } else {
                simToEachTarget = similarityCache.at(simPair);
            }

            if (simToEachTarget > tNodeSim) {
                tNodeSim = simToEachTarget;
                tNodeBest = tNode.first;
            }

            // For target nodes, the document is themselves (the target sentence)
            // the corpus is the preset corpus
            float tfidf = isCorpusSet ?
                          computeTfidf(tNode.first,
                                       computeNodeCountSum(targetNodes),
                                       float(corpusSize),
                                       targetNodes,
                                       documentCountOfNodeInCorpus) : 1;

#ifdef DEBUG_DECISION
            cout << fmt::format("Target node [{}:{}], node [{}:{}], sim {}, tfidf {}",
                                                kb.nodes[tNode.first], tNode.first,
                                                kb.nodes[node], node,
                                                simToEachTarget, tfidf) << endl;
#endif

            recall += simToEachTarget * tfidf;
            recall_idf_sum += tfidf;
        }

#ifdef DEBUG_DECISION
        cout << fmt::format("Node [{}:{}], most similar [{}:{}], sim {}, tfidf {}",
                                            kb.nodes[node], node,
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
    float betaSquare = beta * beta;
    return ((1 + betaSquare) * recall * precision) / (recall + betaSquare * precision + 1e-6);
}

float KnowledgeMatcher::computeNodeCountSum(const unordered_map<long, float> &nodeCount) {
    float sum = 0;
    for (auto &nc : nodeCount)
        sum += nc.second;
    return sum;
}

float KnowledgeMatcher::computeTfidf(long node, float documentSize, float corpusSize,
                                     const unordered_map<long, float> &nodeCountInDocument,
                                     const unordered_map<long, float> &nodeCountInCorpus) {
    float ncic = 1;
    if (nodeCountInCorpus.find(node) != nodeCountInCorpus.end())
        ncic += nodeCountInCorpus.at(node);

    float idf = log(corpusSize / ncic);

    float countSum = 0;
    float tf = nodeCountInDocument.at(node) / documentSize;
    return tf * idf;
}

void KnowledgeMatcher::matchForSourceAndTarget(const vector<int> &sourceSentence,
                                               const vector<int> &targetSentence,
                                               const vector<int> &sourceMask,
                                               const vector<int> &targetMask,
                                               vector<pair<size_t, vector<int>>> &sourceMatch,
                                               vector<pair<size_t, vector<int>>> &targetMatch,
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

void KnowledgeMatcher::normalizeMatch(vector<pair<size_t, vector<int>>> &match,
                                      const vector<int> &mask,
                                      size_t position,
                                      const vector<int> &node,
                                      size_t splitNodeMinimumEdgeNum,
                                      float splitNodeMinimumSimilarity) const {
    long nodeId = kb.nodeMap.at(node);
    auto sizes = kb.getInAndOutEdgeNumOfNode(nodeId, true);
    if (get<0>(sizes) + get<1>(sizes) < splitNodeMinimumEdgeNum) {
//#ifdef DEBUG
//        cout << fmt::format("Splitting node [{}:{}]", kb.nodes[nodeId], nodeId) << endl;
//#endif
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
                        match.emplace_back(make_pair(position + subMatch.first, subSubMatch));
//#ifdef DEBUG
//                        cout << fmt::format("Splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
//#endif
                    } else {
//#ifdef DEBUG
//                        cout << fmt::format("Ignore splitted node [{}:{}]", kb.nodes[subNodeId], subNodeId) << endl;
//#endif
                    }
                }
            }
        }
    } else {
        match.emplace_back(make_pair(position, node));
    }
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
