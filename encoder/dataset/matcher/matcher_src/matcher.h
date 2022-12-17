#ifndef MATCHER_H
#define MATCHER_H
// Uncomment below macro to enable viewing the decision process
//#define DEBUG_DECISION
//#define DEBUG
#include "cista.h"
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"
#include "highfive/H5File.hpp"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// source id, relation id, target id, weight, annotated representation.
using Edge = std::tuple<long, long, long, float, std::string>;

class TrieNode {
  public:
    int token = -1;
    bool isWordEnd = false;
    std::unordered_map<int, TrieNode *> children;

  public:
    TrieNode() = default;

    TrieNode(int token, bool isWordEnd);

    ~TrieNode();

    TrieNode *addChild(int childToken, bool isChildWordEnd);

    bool removeChild(int childToken);

    void clear();
};

class Trie {
  public:
    TrieNode root;

  public:
    Trie() = default;

    explicit Trie(const std::vector<std::vector<int>> &words);

    void insert(const std::vector<int> &word);

    bool remove(const std::vector<int> &word);

    void clear();

    std::vector<std::vector<int>> matchForStart(const std::vector<int> &sentence,
                                                size_t start = 0,
                                                bool allowSubMatch = false) const;

    std::unordered_map<size_t, std::vector<std::vector<int>>>
    matchForAll(const std::vector<int> &sentence, bool allowSubMatch = false) const;
};

class KnowledgeBase {
  public:
    struct SerializableEdge {
        long source;
        long relation;
        long target;
        float weight;
        cista::raw::string annotation;
    };

    struct KnowledgeArchive {
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeToTarget;
        cista::raw::hash_map<long, cista::raw::vector<size_t>> edgeFromSource;
        cista::raw::vector<SerializableEdge> edges;
        cista::raw::vector<cista::raw::string> nodes;
        cista::raw::vector<cista::raw::string> relationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedNodes;
        cista::raw::vector<cista::raw::vector<int>> tokenizedRelationships;
        cista::raw::vector<cista::raw::vector<int>> tokenizedEdgeAnnotations;
        cista::raw::vector<cista::raw::string> rawRelationships;
        cista::raw::string nodeEmbeddingFileName;
    };

    struct VectorHash {
        std::size_t operator()(std::vector<int> const &vec) const;
    };

  public:
    // target id, edge ids point to the target.
    std::unordered_map<long, std::vector<size_t>> edgeToTarget;
    // source id, edge ids point from the source.
    std::unordered_map<long, std::vector<size_t>> edgeFromSource;
    // source id, relation id, target id, weight, annotated representation.
    std::vector<Edge> edges;
    std::vector<bool> isEdgeDisabled;

    std::vector<std::string> relationships;
    std::vector<std::string> rawRelationships;

    std::vector<std::string> nodes;
    Trie nodeTrie;
    std::unordered_map<std::vector<int>, long, VectorHash> nodeMap;
    std::vector<bool> isNodeComposite;
    std::unordered_map<long, std::unordered_map<long, float>> compositeNodes;
    std::unordered_map<long, float> compositeComponentCount;
    std::shared_ptr<void> nodeEmbedding;
    std::string nodeEmbeddingFileName;
    size_t nodeEmbeddingDim = 0;
    bool nodeEmbeddingSimplifyWithInt8 = false;

    std::vector<std::vector<int>> tokenizedNodes;
    std::vector<std::vector<int>> tokenizedRelationships;
    std::vector<std::vector<int>> tokenizedEdgeAnnotations;

  public:
    KnowledgeBase() = default;

    void clearDisabledEdges();

    void disableAllEdges();

    void disableEdgesWithWeightBelow(float minWeight);

    void disableEdgesOfRelationships(const std::vector<std::string> &relationships,
                                     bool excludeCompositeNodes = true);

    void enableEdgesOfRelationships(const std::vector<std::string> &relationships,
                                    bool excludeCompositeNodes = true);

    void disableEdgesOfNodes(const std::vector<std::string> &nodes);

    std::vector<long> findNodes(const std::vector<std::string> &nodes,
                                bool quiet = false) const;

    std::vector<Edge> findEdges(const std::vector<std::string> &source_nodes,
                                const std::vector<std::string> &relations,
                                const std::vector<std::string> &target_nodes,
                                bool quiet = false) const;

    std::unordered_set<long> getNodeNeighbors(long node) const;

    std::vector<Edge> getEdges(long source = -1, long target = -1) const;

    std::vector<Edge> getEdgesBidirection(long node1, long node2) const;

    const std::vector<std::string> &getNodes() const;

    std::vector<std::string> getNodes(const std::vector<long> &nodeIndexes) const;

    std::tuple<size_t, size_t>
    getInAndOutEdgeNumOfNode(long node, bool excludeCompositeNodes = true) const;

    long getCompositeStart() const;

    long addCompositeNode(const std::string &compositeNode,
                          const std::string &relationship,
                          const std::vector<int> &tokenizedCompositeNode,
                          const std::vector<int> &mask = {},
                          const std::vector<int> &connectionMask = {},
                          size_t splitNodeMinimumEdgeNum = 20,
                          float splitNodeMinimumSimilarity = 0.35);

    size_t addCompositeEdge(long sourceNodeId, long relationId, long compositeNodeId);

    void setNodeEmbeddingFileName(const std::string &path);

    int bfsDistance(long node1, long node2, int maxDepth = 3) const;

    bool isNeighbor(long node1, long node2) const;

    float cosineSimilarity(long node1, long node2) const;

    void save(const std::string &archivePath) const;

    void load(const std::string &archivePath);

    void refresh();

  private:
    struct PriorityCmp {
        template <typename T1, typename T2>
        bool operator()(const std::pair<T1, T2> &pair1, const std::pair<T1, T2> &pair2);
    };

  private:
    // For faster evaluation of function isNeighbor() and distance()
    std::unordered_map<long, std::unordered_set<long>> adjacency;
    std::unordered_map<std::string, long> nodeToIndex;

  private:
    void loadEmbedding();

    void loadAdjacency();

    void loadNodeToIndex();

    template <typename T>
    static cista::raw::vector<T> vector2ArchiveVector(const std::vector<T> &vec);

    template <typename T>
    static std::vector<T> archiveVector2Vector(const cista::raw::vector<T> &vec);
};

class KnowledgeMatcher {
  public:
    struct PairHash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &pair) const;
    };

    struct VisitedPath {
        int round;
        long root;
        size_t rootStartPos;
        size_t rootEndPos;
        std::vector<size_t> edges;
        std::unordered_map<size_t, float> similarities;
        std::unordered_set<long> visitedNodes;

        float bestSimilarity;
        std::vector<size_t> uncoveredEdges;
    };

    struct VisitedSubGraph {
        std::vector<VisitedPath> visitedPaths;
        std::unordered_set<long> coveredCompositeNodes;
        std::unordered_set<std::pair<long, long>, PairHash> coveredNodePairs;
        // (source node start pos, source node end pos), edges
        std::unordered_map<std::pair<size_t, size_t>, std::vector<size_t>, PairHash>
            coveredSubGraph;
    };

    typedef std::tuple<std::vector<std::vector<std::vector<int>>>,
                       std::vector<std::vector<Edge>>, std::vector<std::vector<long>>>
        PathResult;
    typedef std::tuple<std::vector<long>, std::vector<long>> SourceAndTargetNodes;
    typedef std::tuple<std::vector<std::vector<std::vector<int>>>,
                       std::vector<std::vector<long>>, std::vector<long>,
                       std::vector<std::vector<Edge>>>
        ChoiceResult;

  public:
    KnowledgeBase kb;
    bool isCorpusSet = false;
    size_t corpusSize = 0;
    std::unordered_map<long, float> documentCountOfNodeInCorpus;

  public:
    explicit KnowledgeMatcher(const KnowledgeBase &knowledgeBase);

    explicit KnowledgeMatcher(const std::string &archivePath);

    void setCorpus(const std::vector<std::vector<int>> &corpus);

    float computeFBetaScore(long node, const std::vector<long> &targetNodes,
                            float beta = 2) const;

    std::string findClosestConcept(std::string targetConcept,
                                   const std::vector<std::string> &concepts) const;

    PathResult findShortestPath(const std::vector<int> &sourceSentence,
                                const std::vector<int> &targetSentence,
                                const std::vector<std::string> &intermediateNodes,
                                const std::vector<int> &sourceMask = {},
                                const std::vector<int> &targetMask = {},
                                bool findTarget = false, size_t maxDepthForEachNode = 3,
                                size_t minLevelsBeforeCheckingTargetReached = 0,
                                size_t splitNodeMinimumEdgeNum = 20,
                                float splitNodeMinimumSimilarity = 0.35) const;

    ChoiceResult findAvailableChoices(const std::vector<long> &visitedNodes,
                                      const std::vector<long> &startNodes,
                                      const std::vector<long> &targetNodes,
                                      const std::vector<long> &allowedCompositeNodes = {},
                                      int maxDepth = 2, bool parallel = true,
                                      bool findTarget = false, bool findComposite = true,
                                      bool filterCompositeNodesByFBeta = false,
                                      float minimumFBeta = 0) const;

    std::vector<std::vector<std::vector<int>>>
    subPathsToAnnotations(const std::vector<std::vector<Edge>> &subPaths) const;

    std::vector<std::vector<std::string>>
    subPathsToStringAnnotations(const std::vector<std::vector<Edge>> &subPaths,
                                const std::vector<std::string> &relationshipTemplates,
                                bool prioritizeOriginalAnnotation = true,
                                bool lowerCase = true) const;

    SourceAndTargetNodes matchSourceAndTargetNodes(
        const std::vector<int> &sourceSentence, const std::vector<int> &targetSentence,
        const std::vector<int> &sourceMask = {}, const std::vector<int> &targetMask = {},
        size_t splitNodeMinimumEdgeNum = 20,
        float splitNodeMinimumSimilarity = 0.35) const;

    void save(const std::string &archivePath) const;

    void load(const std::string &archivePath);

  private:
    std::vector<int> edgeToAnnotation(size_t edgeIndex) const;

    std::string edgeToStringAnnotation(size_t edgeIndex) const;

    std::vector<int> edgeToAnnotation(const Edge &edge) const;

    std::string edgeToStringAnnotation(const Edge &edge) const;

    static size_t
    componentIntersection(const std::unordered_map<long, float> &sourceNodes,
                          const std::unordered_map<long, float> &targetNodes);

    float computeFBetaScoreWithCache(
        long node, const std::unordered_map<long, float> &targetNodes,
        std::unordered_map<std::pair<long, long>, float, PairHash> &similarityCache,
        float beta = 2) const;

    static float computeNodeCountSum(const std::unordered_map<long, float> &nodeCount);

    static float computeTfidf(long node, float documentSize, float corpusSize,
                              const std::unordered_map<long, float> &nodeCountInDocument,
                              const std::unordered_map<long, float> &nodeCountInCorpus);

    void matchForSourceAndTarget(
        const std::vector<int> &sourceSentence, const std::vector<int> &targetSentence,
        const std::vector<int> &sourceMask, const std::vector<int> &targetMask,
        std::vector<std::pair<size_t, std::vector<int>>> &sourceMatch,
        std::vector<std::pair<size_t, std::vector<int>>> &targetMatch,
        size_t splitNodeMinimumEdgeNum, float splitNodeMinimumSimilarity) const;

    void normalizeMatch(std::vector<std::pair<size_t, std::vector<int>>> &match,
                        const std::vector<int> &mask, size_t position,
                        const std::vector<int> &node, size_t splitNodeMinimumEdgeNum,
                        float splitNodeMinimumSimilarity) const;

    static void keepTopK(std::vector<float> &weights, int k = -1);
};

#endif // MATCHER_H
