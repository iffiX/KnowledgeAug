//
// Created by iffi on 2021/10/8.
//
#include <fstream>
#include <iostream>
#include "matcher.h"
#include "concept_net.h"
using namespace std;
#include <ctime>
#define TIME_BEGIN(timer_num) \
    struct timeval timer_ ##timer_num## _start;\
    gettimeofday(&timer_ ##timer_num ##_start, NULL);

#define TIME_END(timer_num) \
    struct timeval timer_ ##timer_num## _end;\
    gettimeofday(&timer_ ##timer_num## _end, NULL);\
    {\
        double msec = (double)(timer_ ##timer_num## _end.tv_usec - timer_ ##timer_num## _start.tv_usec) / 1000 +\
                      (double)(timer_ ##timer_num## _end.tv_sec - timer_ ##timer_num## _start.tv_sec) * 1000;\
        printf(">>> timer_" #timer_num ":    %.3lf ms\n", msec);\
    }


template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<std::vector<T>>& vec) {
    if ( !vec.empty() ) {
        out << '[';
        for(auto &subVec : vec) {
            cout << '[';
            for (T v: subVec)
                out << v << ", ";
            out << "\b\b]";
        }
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    else {
        out << "empty";
    }
    return out;
}

template <typename T1, typename T2>
std::ostream& operator<< (std::ostream& out, const std::tuple<T1, T2>& tuple) {
    out << '(' << std::get<0>(tuple) << ", "<< std::get<1>(tuple) << ')';
    return out;
}

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::unordered_map<size_t, T>& map) {
    if ( !map.empty() ) {
        out << '{';
        for(auto &item : map)
            out << item.first << ": "<< item.second << ", ";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    else {
        out << "empty";
    }
    return out;
}

int main() {
    auto node = TrieNode();
    // pass
    //node.initializeFromString("1,([2$])");

    // err
    //node.initializeFromString("1,([2$],[3,([4$])]");

//    node.initializeFromString("1,([2$],[3$,([5$],[6$])])");
//    cout << node.serialize() << endl;
//
//    node.initializeFromString("1,([2$],[3,([4$],),],)");
//    cout << node.serialize() << endl;
//    node.initializeFromString("1,([2$],[3,([4$],[5,([6$],[7$])])])");
//    cout << node.serialize() << endl;
//
//    auto trie = Trie({{1, 2}, {1, 3, 4}, {1, 3}, {1, 3, 5, 6}, {1, 3, 5, 7}});
//    cout << trie.serialize() << endl;
//
//    trie.initializeFromString("-1,([1,([3,([5$,([7$],[6$])],[4$])],[2$])])");
//    cout << trie.serialize() << endl;
//    trie.insert({1,3,5,7,8});
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3,5,7,8}) << endl;
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3}) << endl;
//    cout << trie.serialize() << endl;
//    cout << trie.remove({1,3, 5}) << endl;
//    cout << trie.serialize() << endl;
//    trie.insert({1, 3, 5});
//    cout << trie.serialize() << endl;

//    auto trie = Trie({{2}, {3, 4}, {4, 1}, {1, 2}, {1, 3}, {1, 3, 4}, {1, 3, 5, 6}, {1, 3, 5, 7}});
//    cout << trie.serialize() << endl;
//    cout << trie.matchForStart({1, 2, 18, 32}) << endl;
//    cout << trie.matchForStart({3, 1, 2, 18, 32}) << endl;
//    cout << trie.matchForStart({123, 1, 2, 18, 32}) << endl;
//    cout << trie.matchForAll({3, 1, 2, 18, 32}) << endl;
    // 3, 4, 1 for checking overlap match
    // 1, 3, 5 for checking partial match (1, 3, and 1, 3, 5, 6)
//    cout << trie.matchForAll({9, 3, 4, 1, 3, 5, 1, 2, 8, 16, 1, 3, 4, 9}) << endl;

//    ConceptNetReader kb("/data/conceptnet-assertions-5.7.0.csv");
//    cout << "Node num: " << kb.nodes.size() << endl;
//    cout << "Relation num: " << kb.relationships.size() << endl;
//    cout << "Raw relation num: " << kb.rawRelationships.size() << endl;
//    cout << "Edge num: " << kb.edges.size() << endl;
//    cout << "Relations: " << kb.relationships << endl;

//    ConceptNetReader kb("/data/conceptnet-assertions-5.7.0_slice.csv");
//    ConceptNetMatcher matcher(kb);
//    matcher.save("/data/conceptnet-archive.data");
//      ConceptNetMatcher matcher2("/data/conceptnet-archive.data");
//      cout << matcher2.getNodeTrie() << endl;

//    Trie trie;
//    ifstream stream("/data/conceptnet-trie.txt");
//    string line;
//    getline(stream, line);
//    trie.initializeFromString(line);
//    cout << trie.serialize() << endl;


//    ConceptNetReader reader;
//    auto kb = reader.read("/data/conceptnet-assertions-5.7.0_slice.csv",
//                "/data/numberbatch-en.txt",
//                "/data/embedding.hdf5");

//    std::vector<int> sentence{71, 3, 60, 4571, 3745, 1365, 19, 4979, 21, 192, 2212, 1111, 6, 68, 34, 92, 4657, 38, 3, 9, 1034, 3613, 44, 3, 9, 125, 58, 4739, 6, 31866, 6, 221, 2274, 297, 1078, 6, 1982, 40, 6, 5534, 25453};
//    KnowledgeMatcher matcher("/data/conceptnet-archive.data");
//    cout << matcher.matchByToken(sentence, {}, 1000, 2, 5, 1920301, 0, {{18097}, {2187, 24302}}, {{1996}, {1037}, {2019}, {2002}, {2000}, {2010}}, {}) << endl;


//    std::vector<int> sourceSentence{2924, 1010, 3075, 1010, 2533, 3573, 1010, 6670, 1010, 2047, 2259};
//    std::vector<int> sourceMask{1,0,0,0,1,1,0,1,0,0,1};
//    std::vector<int> targetSentence{1037,24135,2341,2003,14057,2005,2048,3257,3604,1010,2021,2009,2036,4240,2004,
//                                    1037,3036,5468,2012,1037,2054,1029};
//    std::vector<int> targetMask{0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0};
//    KnowledgeMatcher matcher("/data/conceptnet-archive.data");

//    matcher.matchByNodeEmbedding(sourceSentence, targetSentence, sourceMask, targetMask,
//                                 300, 2, 12,1920301,0.5,0);

//    std::vector<int> sourceSentence{2065,1037,13774,2003,10768,1010,2008,13774,2003,1029};
//    std::vector<int> sourceMask{0,0,1,0,1,0,0,1,0,0};
//    std::vector<int> targetSentence{3707,13774,2065,1037,13774,2003,10768,1010,2008,13774,2003,1029};
//    std::vector<int> targetMask{1,1,0,0,1,0,1,0,0,1,0,0};
//    //std::vector<std::vector<int>> rankFocus{{2242,2017}};
    KnowledgeMatcher matcher("/home/muhan/data/workspace/kb_encoder/data/preprocess/conceptnet-archive.data");
    //auto nodes = matcher.kb.findNodes({"child", "organism"});
    //cout << matcher.kb.cosineSimilarity(nodes[0], nodes[1]);
//    matcher.findAvailableChoices({4760,1165772}, {379995,24647,4169}, {1248,4709});
//    // matcher.matchByNodeEmbedding(sourceSentence, targetSentence, sourceMask, targetMask, 1000, 3, 697474, 10, 1, true, 0);
    matcher.kb.addCompositeNode("detailed observation of celestial objects requires a telescope",
                                "RelatedTo",
                                std::vector<int>{6851, 8089, 1997, 17617, 5200, 5942, 1037, 12772},
                                std::vector<int>{1, 1, 0, 1, 1, 1, 0, 1});
    matcher.findShortestPath(
        std::vector<int>{2478,13536,2000,3579,5067,2422,2013,16581,4230,4473,1029},
        std::vector<int>{6851,8089},
        std::vector<std::string>{"detailed observation of celestial objects requires a telescope"},
        std::vector<int>{1,1,0,1,1,1,0,1,1,1,0},
        std::vector<int>{1,1},
        3,
        0
    );

//    matcher.matchByToken(sourceSentence, targetSentence, sourceMask, targetMask,
//                                 300, 2, 12,1920301,3,0.5,0, rankFocus, {});

    //    KnowledgeMatcher matcher("/data/conceptnet-archive.data");
//    matcher.kb.initLandmarks(100, 10, -1, "/data/conceptnet-landmark.cache");
//    // bank -> security, distance = 1
//    TIME_BEGIN(0);
//    for(int i = 0; i < 1000; i++)
//        matcher.kb.distance(2037, 9779);
//    TIME_END(0);
//    cout << matcher.kb.distance(2037, 9779) << endl;
//    // bank -> security measure, distance = 3
//    TIME_BEGIN(1);
//    for(int i = 0; i < 100; i++)
//        matcher.kb.distance(2037, 904700);
//    TIME_END(1);
//    cout << matcher.kb.distance(2037, 904700) << endl;
//    //cout << matcher.matchByNode(sentence, targetSentence, {}, {}, 300, 2, 5, 1920301, 0);
}
