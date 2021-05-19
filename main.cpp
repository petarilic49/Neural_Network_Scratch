#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream> //Used to convert the string values into integer values
#include <fstream> //Input/output stream class to operate on files


using namespace std;

// Function reads the .csv file and stores/returns the data inputs in a 2D vector
vector<vector<string>> readDataFile(string fileName){

    //Input the heart data into the program 
    ifstream dataFile;
    dataFile.open(fileName);
    vector<vector<string>> dataEntries;
    string inputs[14];
    vector<string> rowData;
    while(getline(dataFile, inputs[0], ',')){
        for(int i = 1; i<14; i++){
            if(i != 13){
                getline(dataFile, inputs[i], ',');
            }
            else{
                getline(dataFile, inputs[i], '\n');
            }
        }
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

void readDataInputs(vector<vector<double>> datainputs){
    for(int i = 0; i<datainputs.size(); i++){
        for(int j = 0; j<datainputs[i].size(); j++){
            cout << datainputs[i][j] << " ";
        }
        cout << endl;
    }
    return;
}

void splitData(vector<vector<double>> totalData, vector<vector<double>> *xtraining, vector<vector<double>> *ytraining, vector<vector<double>> *xtesting, vector<vector<double>> *ytesting){
    
    vector<double> xrowdata, yrowdata;
    
    for(int i = 0; i<totalData.size(); i++){
        if(i<round(totalData.size() * 0.7)){
            for(int j = 0; j<totalData[i].size(); j++){
                if(j<13){
                    xrowdata.push_back(totalData[i][j]);
                }
                else{
                    yrowdata.push_back(totalData[i][j]);
                }
                
            }
            xtraining->push_back(xrowdata);
            xrowdata.clear();
            ytraining->push_back(yrowdata);
            yrowdata.clear();
        }
        else{
            for(int j = 0; j<totalData[i].size(); j++){
                if(j<13){
                    xrowdata.push_back(totalData[i][j]);
                }
                else{
                    yrowdata.push_back(totalData[i][j]);
                }
                
            }
            xtesting->push_back(xrowdata);
            xrowdata.clear();
            ytesting->push_back(yrowdata);
            yrowdata.clear();
        }
    }
    return;
}

//Function that will convert the 2D vector of strings into 2D vector of ints
vector<vector<double>> string_to_double(vector<vector<string>> d){
    cout << "In conversion function" << endl;
    vector<vector<double>> convertedData;
    vector<double> rowHolder;
    for(int i = 1; i<d.size(); i++){ // Starting at 1 because the first row is the name of the columns which we dont care about
        for(int j = 0; j<d[i].size(); j++){ 
            stringstream valHolder((d[i][j]));
            double x = 0;
            valHolder >> x;
            rowHolder.push_back(x);
        }
        convertedData.push_back(rowHolder);
        rowHolder.clear();
    }
    return convertedData;
}

// Function that will normalize our input data
void normalizeData(vector<vector<string>> *x_training, vector<vector<string>> *x_testing){
    
    return;
}

int main()
{
    //Input the heart data into the program 
    vector<vector<string>> sdata;
    sdata = readDataFile("heart.csv");
    vector<vector<double>> data;
    data = string_to_double(sdata);

  
    //readDataInputs(data);
    vector<vector<double>> x_trainingData, y_trainingData, x_testingData, y_testingData;

    splitData(data, &x_trainingData, &y_trainingData, &x_testingData, &y_testingData);

    

    // Next I want to split the data into training and testing data

    return 0;
}