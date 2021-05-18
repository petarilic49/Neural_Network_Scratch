#include <iostream>
#include <vector>
#include <string>
#include <fstream> //Input/output stream class to operate on files


using namespace std;

/*vector<vector<int>> create2DVector(){
    int row;
    row = 6;
    int col;
    col = 5;
    int count = 0;
    vector<vector<int>> retData(row);

    for(int i = 0; i<row; i++){
        retData[i] = vector<int>(col);
        for(int j = 0; j<col; j++){
            retData[i][j] = count;
            count++;
        }
    }

    return retData;
}*/
vector<vector<string>> readDataFile(string fileName){
    // Find the number of rows in the csv file
    int row = 0;

    //Input the heart data into the program 
    ifstream dataFile;
    dataFile.open(fileName);
    string line;
    while(getline(dataFile, line)){
        row++;
    }
    string test;
    getline(dataFile, test, ',');
    cout << test << endl;
    cout << "The Number of Rows in the File is: " << row << endl;
    // Create a 2D vector to hold the rows and columns
    vector<vector<string>> dataEntries;
    
    for(int i = 0; i<row; i++){
        vector<string> rowData;
        for(int j = 0; j<14; j ++){
            string holder;
            getline(dataFile, holder, ',');
            rowData.push_back(holder);
        }
        dataEntries.push_back(rowData);
    }
    dataFile.close();
    cout << "Done Reading the heart file" << endl;
    return dataEntries;
}

int main()
{
    //Input the heart data into the program 
    vector<vector<string>> data;
    data = readDataFile("heart.csv");

    cout << "Size of the vector is: " << data.size() << endl;
    /*for(int i = 0; i<data.size(); i++){
        for(int j = 0; j<data[i].size(); j++){
            cout << data[i][j] << " ";
        }
        cout << endl;
    }*/
    //cout << "First Element is: " << data[0] << endl;
    //cout << "Last Element is: " << data[data.size()] << endl;
    /*for (int row = 0; row<data.size() / 14; row++){
        for (int col = 0; col<15;col++){
            cout << data[row + col] << " " ;
        }
        cout << endl;
    }*/
    

    return 0;
}