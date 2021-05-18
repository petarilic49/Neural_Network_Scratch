#include <iostream>
#include <vector>
#include <fstream> //Input/output stream class to operate on files


using namespace std;

int main()
{
    //Input the heart data into the program 
    ifstream dataFile;
    dataFile.open("heart.csv");

    while(dataFile.good()){
        string line;
        getline(dataFile, line, ',');
        cout << line << endl;
    }

}