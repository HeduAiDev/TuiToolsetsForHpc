#pragma once

#include <vector>
#include <queue>
#include <unordered_map>

namespace utils {
    using namespace std;

    struct pair_hash {
        template <class T1, class T2>
        size_t operator () (const std::pair<T1,T2> &p) const {
            auto h1 = hash<T1>{}(p.first);
            auto h2 = hash<T2>{}(p.second); 
            return h1 ^ h2;
        }
    };
    template <typename T>
    using pair_map = std::unordered_map<std::pair<int, int>, T, pair_hash>;
    template <typename T>
    using label_map = std::unordered_map<int, T>;
    template <typename T>
    T& row_major_vect_get(std::vector<vector<T>>& vec, int x, int y, T default = nullptr) {
        if (y < 0 || y >= vec.size() || x < 0 || x >= vec[y].size()) {
            return default;
        }
        return vec[y][x];
    }


    template<typename ... Args>
    static std::string str_format(const std::string &format, Args ... args)
    {
        auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; 
        std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

        if (!buf)
            return std::string("");

        std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
        return std::string(buf.get(), buf.get() + size_buf - 1); 
    }

    // std::wstring
    template<typename ... Args>
    static std::wstring wstr_format(const std::wstring &format, Args ... args)
    {
        auto size_buf = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; 
        std::unique_ptr<char[]> buf(new(std::nothrow) char[size_buf]);

        if (!buf)
            return std::wstring("");

        std::snprintf(buf.get(), size_buf, format.c_str(), args ...);
        return std::wstring(buf.get(), buf.get() + size_buf - 1); 
    }
}
