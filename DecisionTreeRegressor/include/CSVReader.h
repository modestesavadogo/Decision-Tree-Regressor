#ifndef CSVREADER_H
#define CSVREADER_H

#include "Matrix.h"
#include <string>

/**
 * @brief Utility to load a CSV file into a Matrix.
 */
class CSVReader {
public:
    /**
     * @brief Load numerical data from a CSV file.
     * @param filename Path to CSV file.
     * @param skip_header If true, ignore first line.
     * @return Matrix containing the data.
     * @throws std::runtime_error if file cannot be opened.
     */
    static Matrix load(const std::string& filename, bool skip_header = true);
};

#endif // CSVREADER_H