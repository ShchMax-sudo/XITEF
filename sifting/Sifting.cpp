#include "Parser.h"
#include "FenwickTree.h"
#include "Sifting.h"
#include "set"

std::vector<int> calcPhotons(Photon* photons, size_t n, double selectionTime, std::vector<bool> &chosenIndexes, int maxx, int maxy, int localPsf) {
    bool f = selectionTime != 0;
    bool choose = chosenIndexes.size() != 0;
    if (f) {
        selectionTime /= 2.;
    }
    Photon* arr = photons;
    if (f) {
        arr = new Photon[0]();
    }
    FenwickTree ft = FenwickTree(arr, (f) ? (0) : (n), maxx, maxy);
    std::vector<int> result = std::vector<int>(n);
    size_t left = 0;
    size_t right = 0;
    int maxResult = 0;  
    for (size_t i = 0; i < n; ++i) {
        if (choose && !chosenIndexes[i]) {
            continue;
        }
        if (f) {
            double currentTime = photons[i].getTime();
            while (currentTime - photons[left].getTime() > selectionTime) {
                std::pair<int, int> xy = ft.eventToXY(photons[left]);
                ft.add(xy.first, xy.second, -1);
                left++;
            }
            while (right < n && photons[right].getTime() - currentTime <= selectionTime) {
                std::pair<int, int> xy = ft.eventToXY(photons[right]);
                ft.add(xy.first, xy.second, 1);
                right++;
            }
        }
        std::pair<int, int> xy = ft.eventToXY(photons[i]);
        result[i] = ft.count(xy.first - localPsf, xy.first + localPsf, xy.second - localPsf, xy.second + localPsf);
        maxResult = std::max(maxResult, result[i]);
    }
    if (f) {
        delete[] arr;
    }
    return result;
}

py::array_t<int> calc(std::vector<double> &x, std::vector<double> &y, std::vector<double> &time, int maxx, int maxy, int localPsf, std::vector<bool> &flags, double selectionTime) {
    std::pair<Photon*, size_t> photonsN = getPhotons(x, y, time);
    Photon* photons = photonsN.first;
    size_t n = photonsN.second;
    std::vector<int> counts = calcPhotons(photons, n, selectionTime, flags, maxx, maxy, localPsf);
    return vector_as_array_nocopy(counts);
}

bool checkTime(double timem, double times, double timed, double timemin, double timemax) {
    if (timem - timed < timemin) {
        return times - timemin <= 2 * timed;
    }
    if (timem + timed > timemax) {
        return timemax - times <= 2 * timed;
    }
    return abs(times - timem) <= timed;
}

py::array_t<int> count(std::vector<double> &xm, std::vector<double> &ym, std::vector<double> &timem, std::vector<double> &timed, std::vector<double> &xs, std::vector<double> &ys, std::vector<double> &times, double localPsf, double innerSigmaCoefficient, double outerSigmaCoefficient, double timeCoefficient) {
    size_t n = xm.size();
    std::vector<int> cnt = std::vector<int>(2 * n, 0);

    double timemin = times[0];
    double timemax = times[0];

    for (int i = 0; i < times.size(); ++i) {
        if (timemin > times[i]) {
            timemin = times[i];
        }
        if (timemax < times[i]) {
            timemax = times[i];
        }
    }

    for (int j = 0; j < xm.size(); ++j) {
        for (int i = 0; i < xs.size(); ++i) {
            double r = sqrt((xs[i] - xm[j]) * (xs[i] - xm[j]) + (ys[i] - ym[j]) * (ys[i] - ym[j]));
            if (r <= localPsf && checkTime(timem[j], times[i], timed[j], timemin, timemax)) {
                cnt[j] += 1;
            } else if (localPsf * innerSigmaCoefficient < r && r <= localPsf * outerSigmaCoefficient && checkTime(timem[j], times[i], timed[j] * timeCoefficient, timemin, timemax)) {
                cnt[j + n] += 1;
            }
        }
    }
    return vector_as_array_nocopy(cnt);
}

void removeElement(std::multiset<std::pair<double, int>> &probset, std::multiset<std::pair<double, int>> &timeset, int ind, std::vector<double> &prob, std::vector<double> &time) {
    probset.erase({-prob[ind], ind});
    timeset.erase({time[ind], ind});
}

py::array_t<int> clustering(std::vector<double> &x, std::vector<double> &y, std::vector<double> &time, std::vector<double> &prob, int localPsf, int timeCoeff) {
    size_t n = x.size();
    int groupNum = 0;
    int eventsLeft = n;
    std::multiset<std::pair<double, int>> probset;
    std::multiset<std::pair<double, int>> timeset;
    for (int i = 0; i < n; ++i) {
        probset.insert({-prob[i], i});
        timeset.insert({time[i], i});
    }
    std::vector<int> cluster(n);
    while (eventsLeft) {
        int start = probset.begin()->second;
        int left, right, prev;
        left = start;
        right = left;
        prev = left;
        cluster[start] = groupNum;
        removeElement(probset, timeset, start, prob, time);
        eventsLeft--;
        while (true) {
            std::multiset<std::pair<double, int>>::iterator canditer = timeset.upper_bound({time[prev], prev});
            if (canditer == timeset.end()) {
                break;
            }
            int candidate = canditer->second;
            if (time[candidate] - time[right] > timeCoeff) {
                break;
            }
            prev = candidate;
            if ((x[start] - x[candidate]) * (x[start] - x[candidate])
                + (y[start] - y[candidate]) * (y[start] - y[candidate]) <= localPsf * localPsf) {
                cluster[candidate] = groupNum;
                eventsLeft--;
                right = prev;
                removeElement(probset, timeset, prev, prob, time);
            }
        }
        prev = left;
        while (true) {
            std::multiset<std::pair<double, int>>::iterator canditer = timeset.lower_bound({time[prev], prev});
            if (canditer == timeset.begin()) {
                break;
            } else {
                canditer--;
            }
            int candidate = canditer->second;
            if (time[left] - time[candidate] > timeCoeff) {
                break;
            }
            prev = candidate;
            if ((x[start] - x[candidate]) * (x[start] - x[candidate])
                + (y[start] - y[candidate]) * (y[start] - y[candidate]) <= localPsf * localPsf) {
                cluster[candidate] = groupNum;
                eventsLeft--;
                left = prev;
                removeElement(probset, timeset, prev, prob, time);
            }
        }
        groupNum++;
    }

    return vector_as_array_nocopy(cluster);
}