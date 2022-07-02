#ifndef CONCEPT_NET_H
#define CONCEPT_NET_H

#include <memory>
#include <unordered_map>
#include "matcher.h"

class ConceptNetReader {
public:
    KnowledgeBase
    read(const std::string &assertionPath,
         const std::string &weightPath = "",
         const std::string &weightStyle = "numberbatch",
         const std::string &weightHDF5Path = "conceptnet_weights.hdf5",
         bool simplifyWithInt8 = true);

private:
    std::unordered_map<std::string, size_t> weightNames;
    std::unique_ptr<float[]> weights;
    size_t dim = 0;
    Trie weightNameTrie;

    void readWeights(const std::string &weightPath, const std::string &weightStyle);
};

#endif //CONCEPT_NET_H
