#include "Parser.h"
#include <cmath>
#include <unordered_map>

class FenwickTree {
private:
    // Main tree map
    std::unordered_map<long long, int> t_;
    // Max possible x coordinate
    int maxx_ = 0;
    // Max possible y coordinate
    int maxy_ = 0;

    int sum(int x, int y) {
        int result = 0;
        for (int i = x; i > 0; i -= (i & (-i))) {
            for (int j = y; j > 0; j -= (j & (-j))) {
                long long ij = eventToXY(i, j);
                if (this->t_.find(ij) != this->t_.end()) {
                    result += this->t_[ij];
                }
            }
        }
        return result;
    }

    long long eventToXY(int i, int j) {
        return i * this->maxy_ + j;
    }

public:
    std::pair<int, int> eventToXY(Photon event) {
        return std::make_pair((int) round(event.getX()), (int) round(event.getY()));
    }

    FenwickTree(std::vector<Photon> &photons, size_t n, int maxx, int maxy): t_(std::unordered_map<long long, int>()), maxx_(maxx), maxy_(maxy) {
        for (size_t i = 0; i < n; ++i) {
            std::pair<int, int> xy = this->eventToXY(photons[i]);
            this->add(xy.first, xy.second, 1);
        }
    }

    void add(int x, int y, int val) {
        if (x < 0 || y < 0 || x >= this->maxx_ || y >= this->maxy_) {
            return;
        }
        for (int i = x; i <= this->maxx_; i += (i & (-i))) {
            for (int j = y; j < this->maxy_; j += (j & (-j))) {
                long long ij = this->eventToXY(i, j);
                if (this->t_.find(ij) != this->t_.end()) {
                    this->t_[ij] += val;
                } else {
                    this->t_[ij] = val;
                }
            }
        }
    }

    int count(int x1, int x2, int y1, int y2) {
        return this->sum(x2, y2) - this->sum(x2, y1 - 1) - this->sum(x1 - 1, y2) + this->sum(x1 - 1, y1 - 1);
    }
};
