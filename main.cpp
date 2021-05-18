#include <iostream>
#include <vector>
#include <string>
#include <fstream> //Input/output stream class to operate on files


using namespace std;

// Function reads the .csv file and stores/returns the data inputs in a 2D vector
vector<vector<string>> readDataFile(){

    //Input the heart data into the program 
    ifstream dataFile;
    dataFile.open("heart.csv");
    vector<vector<string>> dataEntries;
    string inputs[14];
    vector<string> rowData;
    while(getline(dataFile, inputs[0], ',')){
        getline(dataFile, inputs[1], ',');
        getline(dataFile, inputs[2], ',');
        getline(dataFile, inputs[3], ',');
        getline(dataFile, inputs[4], ',');
        getline(dataFile, inputs[5], ',');
        getline(dataFile, inputs[6], ',');
        getline(dataFile, inputs[7], ',');
        getline(dataFile, inputs[8], ',');
        getline(dataFile, inputs[9], ',');
        getline(dataFile, inputs[10], ',');
        getline(dataFile, inputs[11], ',');
        getline(dataFile, inputs[12], ',');
        getline(dataFile, inputs[13], '\n');
        for(int i = 0; i<14; i++){
            rowData.push_back(inputs[i]);
        }
        dataEntries.push_back(rowData);
        rowData.clear();
        
    }
    dataFile.close();
    cout << "Done Reading the heart file" << endl;
    return dataEntries;
}

int main()
{
    //Input the heart data into the program 
    vector<vector<string>> data;
    data = readDataFile();

    
    for(int i = 0; i<data.size(); i++){
        for(int j = 0; j<data[i].size(); j++){
            cout << data[i][j] << " ";
        }
        cout << endl;
    }
    

    return 0;
}