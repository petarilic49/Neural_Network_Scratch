#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "eigen-3.4-rc1/Eigen/Dense" // Library for matrix algebra
#include <time.h> // Used to create randomly generated weights between 0 and 1
#include <sstream> //Used to convert the string values into integer values
#include <fstream> //Input/output stream class to operate on files


using namespace std;
using Eigen::MatrixXd;

// Create a class that will stores a dense layer
class neural_layer{
    public:
    MatrixXd weights, biases, outputs, costResult;
    int neurons;
        neural_layer(int num_inputs, int num_neurons){ //Constructor
            // Need to randomly generate weights for each input to each neuron
            neurons = num_neurons;
            weights.resize(num_inputs, num_neurons);
            biases.resize(1, num_neurons);
            biases.setZero();
            // Initialize the costResult for the cost function
            costResult.resize(1,1);
            costResult.setZero();
            srand( (unsigned)time( NULL ) );
            for(int i = 0; i<weights.rows(); i++){
                for(int j = 0; j<weights.cols(); j++){
                    weights(i, j) = (double) rand()/RAND_MAX - 0.35; // Added the -0.35 to offset the weights and make sure we have negative weights
                }
            }  
            return;
        }
        void weightedSum(MatrixXd inputs){
            // Creates forward propogation on the layer, ie generates the output by weighted sum * activation function
            outputs.resize(1, neurons);
            outputs = inputs * weights + biases;
            return;
        }
        void ReLUactivation(){
            for(int i = 0; i<outputs.rows(); i++){
                for(int j = 0; j<outputs.cols(); j++){
                    if(outputs(i, j) < 0.0){
                        outputs(i, j) = 0.0;  
                    }
                }
            }
            return;
        }
        void Sigactivation(){
            double hold = 0;
            for(int i = 0; i<outputs.rows(); i++){
                for(int j = 0; j<outputs.cols(); j++){
                    hold = outputs(i, j);
                    outputs(i,j) = 1 / (1 + exp(-1*hold)); 
                }
            }
            return;
        }
        // Since this is a binary classification problem it is best to use a binary cross entropy cost function to for SGD 
        void costFunction (MatrixXd actuals){ // Assuming that we are training the model with a batch size of x (ie actuals has x values and output has x predicted probabilities)
            // Transpose the batch of actual values
            actuals.transpose();
            if(costResult.rows() != 1){// Add a row to the bottom of the matrix to hold the losses of the new inputs/forward propagation
                costResult.conservativeResize(costResult.rows() + 1, costResult.cols());
                costResult.row(costResult.rows() - 1).setZero(); // Initialize the losses as zero to start
            }
            double hold = 0;
            for(int i = 0; i<actuals.cols(); i++){ // Only need one for loop because we know the actuals are transpose to row vector
                hold = hold + ((actuals(0, i)*log(outputs(0, i))) + ((1 - actuals(0,i))*(1 - log(outputs(0,i)))));
                if(i == (actuals.cols() - 1)){
                    costResult(costResult.rows() - 1, 0) = -1.0 * (hold / (i + 1));
                }
            }
            cout << "Done cost function" << endl;
        }
};

// Function reads the .csv file and stores/returns the data inputs in a 2D vector
/*vector<vector<string>> readDataFile(string fileName){

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
}*/

void readDatatoMat (const string fileName, MatrixXd &mat){ // Matrix is passed by reference, therefore it will get changed in the main()
    // First loop through the file to get the number of rows and columns of inputs
    int num_rows = 0, num_cols = 0;
    ifstream dataFile;
    dataFile.open(fileName);
    string line, cell;
    while(getline(dataFile, line)){
        num_rows++;
        stringstream linestream(line); // Essentially creates a stream that holds the strings in 'line' and that way we can use getline later on
        if(num_rows == 1){ // Ensures we only count the number of columns once and not at every row analyzed
            while(getline(linestream, cell, ',')){
                num_cols++;
            }
        }
    }
    dataFile.close(); //We have to close it now because were at the botton of the data file
    cout << "Number of Rows is: " << num_rows << " Number of Columns is: " << num_cols << endl;
    // Second loop to populate the matrix with the data values
    mat.resize(num_rows, num_cols);
    dataFile.open(fileName); //Reopen the csv file from the top
    num_rows = 0;
    while(getline(dataFile, line)){
        num_cols = 0;
        stringstream linestream(line);
        while(getline(linestream, cell, ',')){
            stringstream valHold(cell);
            double x = 0;
            valHold >> x;
            mat(num_rows, num_cols) = x;
            num_cols++;
        }
        num_rows++;
    }
    dataFile.close();
    return;
}

void removeRow(MatrixXd &mat, int rowtoRemove){
    int rowNum = mat.rows() - 1;
    int colNum = mat.cols();

    if(rowtoRemove<rowNum){
        mat.block(rowtoRemove, 0, rowNum - rowtoRemove, colNum) = mat.block(rowtoRemove+1, 0, rowNum - rowtoRemove, colNum);
    }
    mat.conservativeResize(rowNum, colNum);
    return;
}

/*void readDataInputs(vector<vector<double>> datainputs){
    for(int i = 0; i<datainputs.size(); i++){
        for(int j = 0; j<datainputs[i].size(); j++){
            cout << datainputs[i][j] << " ";
        }
        cout << endl;
    }
    return;
}*/

void readData(MatrixXd &d){
    cout << d << endl;
    return;
}

/*void splitData(vector<vector<double>> totalData, vector<vector<double>> *xtraining, vector<vector<double>> *ytraining, vector<vector<double>> *xtesting, vector<vector<double>> *ytesting){
    
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
}*/

void splitData(MatrixXd &td, MatrixXd &xtrain, MatrixXd &ytrain, MatrixXd &xtest, MatrixXd &ytest){
    int numRows = td.rows();
    int numCols = td.cols();
    int trainRows = round(numRows * 0.7);

    xtrain = td.block(0, 0, trainRows, numCols - 1);
    ytrain = td.block(0, numCols-1, trainRows, 1);
    xtest = td.block(trainRows, 0, numRows - trainRows, numCols - 1);
    ytest = td.block(trainRows, numCols - 1, numRows - trainRows, 1);
    return;
}

//Function that will convert the 2D vector of strings into 2D vector of doubles
/*vector<vector<double>> string_to_double(vector<vector<string>> d){
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
}*/

void normalizeData(MatrixXd &m, int rows, int cols){
    int min = 10000;
    int max = 0;
    int colHold = 0;

    while(colHold < cols){
        for(int i = 0; i<rows; i++){
            if(m(i, colHold) < min){
                min = m(i, colHold);
            }
            else if(m(i, colHold) > max){
                max = m(i, colHold);
            }
        }
        for (int i  = 0; i<rows; i++){
            double val;
            val = (m(i, colHold) - min) / (max - min);
            m(i, colHold) = val;
        }
        colHold++;
        max = 0;
        min = 10000;
    }

    return;

}

// Function that will normalize our input data
/*void normalizeData(vector<vector<double>> *dt){
    cout << "Normalizing Data" << endl;
    double min = 10000;
    double max = 0;
    double hold = 0;
    int col = 0;

    while (col < 14){
        for(int i = 0; i<dt->size(); i++){
            if((*dt)[i][col] < min){
                min = (*dt)[i][col];
            }
            else if((*dt)[i][col] > max){
                max = (*dt)[i][col];
            }
        }
        for(int i = 0; i<dt->size(); i++){
            hold = (*dt)[i][col];
            (*dt)[i][col] = ((*dt)[i][col] - min) / (max - min);
        }
        min = 10000;
        max = 0;
        col++;
    }

    cout << "Done normalizing data" << endl;
    return;
}*/

int main()
{
    // 2.0
    MatrixXd data;
    readDatatoMat("heart.csv", data);
    removeRow(data, 0);
    //cout << "Total Data is: " << endl;
    //cout << data << endl;
    MatrixXd x_training, y_training, x_testing, y_testing;
    
    splitData(data, x_training, y_training, x_testing, y_testing);

    normalizeData(x_training, x_training.rows(), x_training.cols());
    normalizeData(x_testing, x_testing.rows(), x_testing.cols());

    MatrixXd in(1, 4);
    in << 1, 2, 3, 4;
    neural_layer hidden_layer1(4, 3);
    cout << "Weights of hidden layer are: " << endl;
    readData(hidden_layer1.weights);
    cout << "Biases of hidden layer are: " << endl;
    readData(hidden_layer1.biases);
    hidden_layer1.weightedSum(in);
    cout << "Output of hidden layer are: " << endl;
    readData(hidden_layer1.outputs);
    hidden_layer1.Sigactivation();
    cout << "Activated hidden layer output: " << endl;
    readData(hidden_layer1.outputs);
    MatrixXd out(3, 1);
    out << 0, 1, 0;
    hidden_layer1.costFunction(out);
    cout << "Cost Function Output: " << endl;
    readData(hidden_layer1.costResult);


    // Because this is a binary clssification problem, we need to use a cross entropy (ie logarithmic loss) loss function to compare the output and actual values
    // Also need to implement the batch and epoch user interface of the neural network 
   

    
    //Input the heart data into the program 
    /*vector<vector<string>> sdata;
    sdata = readDataFile("heart.csv"); // The data is all stored as strings, but we need them as doubles
    vector<vector<double>> data;
    data = string_to_double(sdata); // Convert the dataset to vector of type double

  
    //readDataInputs(data);
    vector<vector<double>> x_trainingData, y_trainingData, x_testingData, y_testingData;
    // Next I want to split the data into training and testing data
    splitData(data, &x_trainingData, &y_trainingData, &x_testingData, &y_testingData);

    //Next want to scale the input data for best practice 
    normalizeData(&x_trainingData);
    normalizeData(&x_testingData);

    // To make the computation of the neural network easier lets convert the vectors into matrices 



    // We want weights for each neuron to be randomly selected within the range of -1 and 1, prevents output/data explosion
    // In terms of biases initialize them as 0, but if you notice that the output is always 0 there may be a 'dead network' which means weights are multiplied by 0

    Matrix<double, Dynamic, Dynamic> x_trainingMat;
    x_trainingMat = vector_to_matrix(&x_trainingData);

    cout << x_trainingMat << endl;*/

    return 0;
}