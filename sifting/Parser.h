#ifndef MY_PARSER_H
#define MY_PARSER_H

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

class Photon {
private:
    double x_;
    double y_;
    double time_;

public:
    double getX() {
        return this->x_;
    }
    double getY() {
        return this->y_;
    }
    double getTime() {
        return this->time_;
    }

    Photon(): x_(0), y_(0), time_(0) {}
    Photon(double x, double y, double time): x_(x), y_(y), time_(time) {}

    friend std::istream &operator>>(std::istream& stream, Photon& v);
    friend std::ostream &operator<<(std::ostream& stream, Photon& v);
};

std::istream& operator>>(std::istream& stream, Photon& v) {
    stream >> v.x_ >> v.y_ >> v.time_;
    return stream;
}

std::ostream& operator<<(std::ostream& stream, Photon& v) {
    stream << v.x_ << " " << v.y_ << " " << v.time_;
    return stream;
}

std::pair<std::vector<Photon>, size_t> getPhotons(std::vector<double> &x, std::vector<double> &y, std::vector<double> &time) {
    size_t n = x.size();
    std::vector<Photon> photons(n);
    for (size_t i = 0; i < n; ++i) {
        photons[i] = Photon(x[i], y[i], time[i]);
    }
    return std::make_pair(photons, n);
}

#endif
