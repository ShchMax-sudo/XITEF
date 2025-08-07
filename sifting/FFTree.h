#include "Parser.h"

class Node {
public:
    std::pair<double, int> time;
    std::pair<double, int> left, right;
    double prob;
    int lefti;
    int righti;
    double maxprob;
    int ind;
    bool destroyed = false;

    Node() {}

    Node(double time_, double prob_, int ind_): time(std::make_pair(time_, 0)), prob(prob_), ind(ind_), lefti(-1), righti(-1) {}
};

class FFTree {
public:
    std::vector<Node> arr;
    int n;
    int ns;

    FFTree(std::vector<std::vector<double>> el, int n_) {
        this->n = n_;
        this->ns = 1;
        while (this->ns < this->n) {
            this->ns <<= 1;
        }
        this->arr = std::vector<Node>(this->ns);
        for (int i = 0; i < n; ++i) {
            this->arr[i] = Node(el[i][0], el[i][1], i);
            if (i && this->arr[i - 1].time.first == this->arr[i].time.first) {
                this->arr[i].time.second = this->arr[i - 1].time.second + 1;
            }
        }
        std::pair<double, int> lastTime = this->arr[n - 1].time;
        for (int i = n; i < ns; ++i) {
            this->arr[i] = Node(lastTime.first, 0, i);
            this->arr[i].time.second = this->arr[i - 1].time.second + 1;
            this->arr[i].destroyed = true;
        }
        for (int i = 1; i < ns; i += 2) {
            int mask = (((i + 1) ^ i) + 1) >> 1;
            this->arr[i].lefti = i ^ (mask >> 1);
            this->arr[i].righti = i ^ (mask >> 1) | mask;
            if (this->arr[i].righti > ns) {
                this->arr[i].righti = -1;
            }
        }
        this->treeUpdate(this->ns - 1);
    }

    int getProb(int node, double prob) {
        if (node == -1) {
            return -1;
        }
        if (this->arr[node].prob == prob) {
            return node;
        }
        if (this->arr[node].maxprob < prob) {
            return -1;
        }
        return std::max(this->getProb(this->arr[node].lefti, prob),
                        this->getProb(this->arr[node].righti, prob));
    }

    void remove(int node, std::pair<double, int> time) {
        if (node == -1) {
            return;
        }
        if (this->arr[node].left > time || this->arr[node].right < time) {
            return;
        }
        if (this->arr[node].time == time) {
            this->arr[node].destroyed = true;
        } else {
            this->remove(this->arr[node].lefti, time);
            this->remove(this->arr[node].righti, time);
        }
        this->update(node);
    }

    int getLater(int node, std::pair<double, int> time) {
        if (node == -1) {
            return -1;
        }
        if (this->arr[node].right < time) {
            return -1;
        }
        int val = this->getLater(this->arr[node].lefti, time);
        if (val != -1) {
            return val;
        }
        if (this->arr[node].time > time && !this->arr[node].destroyed) {
            return node;
        }
        return this->getLater(this->arr[node].righti, time);
    }

    int getEarlier(int node, std::pair<double, int> time) {
        if (node == -1) {
            return -1;
        }
        if (this->arr[node].left > time) {
            return -1;
        }
        int val = this->getEarlier(this->arr[node].righti, time);
        if (val != -1) {
            return val;
        }
        if (this->arr[node].time < time && !this->arr[node].destroyed) {
            return node;
        }
        return this->getEarlier(this->arr[node].lefti, time);
    }

    void print(int node) {
        if (node == -1) {
            return;
        }
        this->print(this->arr[node].lefti);
        std::cout << this->arr[node].prob << " " << this->arr[node].time.first << " " << this->arr[node].time.second << " " << this->arr[node].ind << " " << node << std::endl;
        this->print(this->arr[node].righti);
    }

    void treeUpdate(int node) {
        if (node == -1) {
            return;
        }
        if (this->arr[node].lefti != -1) {
            this->treeUpdate(this->arr[node].lefti);
            this->arr[node].left = this->arr[this->arr[node].lefti].left;
        } else {
            this->arr[node].left = this->arr[node].time;
        }
        if (this->arr[node].righti != -1) {
            this->treeUpdate(this->arr[node].righti);
            this->arr[node].right = this->arr[this->arr[node].righti].right;
        } else {
            this->arr[node].right = this->arr[node].time;
        }
        this->update(node);
    }

    void update(int node) {
        if (this->arr[node].destroyed) {
            this->arr[node].maxprob = 0;
        } else {
            this->arr[node].maxprob = this->arr[node].prob;
        }
        if (this->arr[node].lefti != -1) {
            this->arr[node].maxprob = std::max(this->arr[node].maxprob, this->arr[this->arr[node].lefti].maxprob);
        }
        if (this->arr[node].righti != -1) {
            this->arr[node].maxprob = std::max(this->arr[node].maxprob, this->arr[this->arr[node].righti].maxprob);
        }
    }
};