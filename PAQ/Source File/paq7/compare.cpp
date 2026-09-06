#include <iostream>
#include <fstream>
#include <iterator>
#include <string>

bool compareFiles(const std::string& filename1, const std::string& filename2) {
    std::ifstream f1(filename1, std::ios::binary | std::ios::ate);
    std::ifstream f2(filename2, std::ios::binary | std::ios::ate);

    if (!f1.is_open() || !f2.is_open()) {
        return false;
    }

    if (f1.tellg() != f2.tellg()) {
        return false;
    }

    f1.seekg(0, std::ios::beg);
    f2.seekg(0, std::ios::beg);

    return std::equal(std::istreambuf_iterator<char>(f1),
                      std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(f2));
}

int main() {
    std::string file1, file2;
    std::cin >> file1 >> file2;

    if (compareFiles(file1, file2)) {
        std::cout << "yes\n";
    } else {
        std::cout << "no\n";
    }

    return 0;
}