#include "CSVReader.h"
#include <fstream>
#include <sstream>
#include <iostream>

Matrix CSVReader::load(const std::string& filename, bool skip_header) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<double>> data;
    std::string line;

    if (skip_header)
        std::getline(file, line); // discard header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> row;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (const std::exception& e) {
                std::cerr << "Warning: non-numeric value '" << cell << "' replaced by 0.0\n";
                row.push_back(0.0);
            }
        }
        if (!row.empty())
            data.push_back(row);
    }
    file.close();
    return Matrix(data);
}