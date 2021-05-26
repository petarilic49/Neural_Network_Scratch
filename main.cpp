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
    MatrixXd weights, biases, outputs;
    double costResult;
    int neurons;
        neural_layer(int num_inputs, int num_neurons){ //Constructor
            // Need to randomly generate weights for each input to each neuron
            neurons = num_neurons;
            weights.resize(num_inputs, num_neurons);
            biases.resize(1, num_neurons);
            biases.setZero();
            // Initialize the costResult for the cost function
            costResult = 0;
            // Initialize the outputs of the layer and set to 0
            outputs.resize(1, neurons);
            outputs.setZero();
            //srand(time(NULL)); // For some reason this was making the weights of the output layer identical to the first row of weights in the hidden layer weights matrix
            for(int i = 0; i<weights.rows(); i++){
                for(int j = 0; j<weights.cols(); j++){
                    weights(i, j) = (double)rand()/RAND_MAX - 0.35; // Added the -0.35 to offset the weights and make sure we have negative weights
                }
            }
            return;
        }
        void weightedSum(MatrixXd inputs){
            // Creates forward propogation on the layer, ie generates the output by weighted sum * activation function
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
        // Also since the data set is so small a batch size of 1 will be used ie the gradient algorithm will be a stochastic gradient decent (SGD) 
        void costFunction (MatrixXd actuals){ // Assuming that we are training the model with a batch size of x (ie actuals has x values and output has x predicted probabilities)
            double hold = 0;
            if(actuals(0, 0) == 1){
                    hold = (-1.0 * log(outputs(0,0)));
            }
            else if(actuals(0,0) == 0){
                    hold = (-1.0 * log(1 - outputs(0,0)));
            }
            costResult = hold;
    
            /*double hold = 0;
            for(int i = 0; i<actuals.cols(); i++){ // Only need one for loop because we know the actuals are transpose to row vector
                if(actuals(0, i) == 1){
                    hold = hold + (-1.0 * log(outputs(0,i)));
                }
                else if(actuals(0,i) == 0){
                    hold = hold + (-1.0 * log(1 - outputs(0,i)));
                }
                if(i == (actuals.cols() - 1)){
                    costResult(costResult.rows() - 1, 0) = (hold / (i + 1));
                }
            }*/
        }
};

void backPropagate(neural_layer &out, neural_layer &hidden, MatrixXd actuals, MatrixXd inputs){ // Function is used to backpropagate through the specific layer and update the weights
    // Start with the output layer (ie in out case its just the weights of the output neuron)
    // Three partial derivatives need to be mulitplied (use naming convention as: y_p_x partial derivative of y with respect to x)
    MatrixXd C_p_w((out.weights.rows() * out.weights.cols()) + (hidden.weights.rows() * hidden.weights.cols()), 1); // Gradient vector (ie holds the gradient with respect to each weight)
    C_p_w.setZero();
    double C_p_y = 0; // Only one value because there is only one output from the neural network
    double y_p_z = 0;
    int count = 0;
    C_p_y = -1.0 * (out.outputs(0,0) - actuals(0,0))/((out.outputs(0,0) - 1)*out.outputs(0,0));
    cout << "C_p_y: " << C_p_y << endl;
    for(int i = 0; i<hidden.outputs.cols(); i++){
        // Find the gradient of the first output 
        y_p_z = exp(hidden.outputs(0,i)) / pow((exp(hidden.outputs(0,i)) + 1),2);
        //cout<<"Inputting " << count << " index for the gradient vector" << endl;
        C_p_w(count, 0) = C_p_y * y_p_z * hidden.outputs(0,i);
        //cout << "Gradient is: " << C_p_w(count, 0) << endl;
        count++;
        for(int j = 0; j<hidden.weights.rows(); j++){
            double der = 0;
            if(hidden.outputs(0, i)>0){
                der = 1;
            }else{
                der = 0;
            }
            //cout<<"Inputting " << count << " index for the gradient vector" << endl;
            C_p_w(count, 0) = C_p_y * y_p_z * out.weights(i, 0) * der * inputs(0,j);
            //cout << "Gradient is: " << C_p_w(count, 0) << endl;
            count++;
        }
    }
    //cout << "The gradient vector is: " << endl;
    //cout << C_p_w << endl;
   
    // Chunk that update the weights of the layers
    double learn_rate = 0.01;
    count = 0;
    for(int i = 0; i<out.weights.rows(); i++){
        out.weights(i, 0) = out.weights(i, 0) - learn_rate*C_p_w(count, 0);
        //cout << " to new weight: " << out.weights(i, 0) << endl;
        count++;
        for(int j = 0; j<hidden.weights.rows(); j++){
            //cout << "Changing old weight: " << hidden.weights(j, i);
            hidden.weights(j, i) = hidden.weights(j, i) - learn_rate*C_p_w(count, 0);
            //cout << "To new weight: " << hidden.weights(j, i) << endl;
            count++;
        }
    }
    return;
}

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
    //cout << "Number of Rows is: " << num_rows << " Number of Columns is: " << num_cols << endl;
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

    // Create a matrix that is randomly sorted with the total data set
    MatrixXd random(numRows, numCols);
    // Keep track which rows have been populated
    vector<int> rowCount;

    int rowindex = 0;
    for(int i = 0; i<numRows; i++){
        rowindex = rand() % numRows; // Random number between 0 and number of rows in data - 1
        // Check if the row index has already been called upon
        int j = 0;
        for(j; j<rowCount.size(); j++){
            if(rowCount[j] == rowindex){ // If that row has already been called then regenerate and restart the checking process
                rowindex = rand() % numRows;
                j = 0;
            }
        }
        rowCount.push_back(rowindex);
        random.row(i) = td.row(rowindex); 
    }

    xtrain = random.block(0, 0, trainRows, numCols - 1);
    ytrain = random.block(0, numCols-1, trainRows, 1);
    xtest = random.block(trainRows, 0, numRows - trainRows, numCols - 1);
    ytest = random.block(trainRows, numCols - 1, numRows - trainRows, 1);

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

    //Create the hidden layer and the output layer
    neural_layer hidden_layer(x_training.cols(), 13);
    neural_layer out_layer(13, 1);
    
    // Ask the user what epoch size they would like (for education purposes were gonn test to see which is the best that reaches the lowest cost function output)
    // Need a double for loop, outer for the epoch, inner to go through the whole training dataset

    /*for(int epoch = 0; epoch<1; epoch++){
        for(int i = 0; i<170; i++){
            // Forward propagate the neural network
            //cout << "Input to hidden layer: " << x_training.row(i) << endl;
            hidden_layer.weightedSum(x_training.row(i));
            hidden_layer.ReLUactivation();
            //cout << "Hidden Layer Output is: " << hidden_layer.outputs << endl;
            //cout << "Input to output layer: " << hidden_layer.outputs << endl;
            out_layer.weightedSum(hidden_layer.outputs);
            cout << "Output Layer Weights: " << out_layer.weights << endl;
            cout << "Output Layer Weighted Sum: " << out_layer.outputs << endl;
            out_layer.Sigactivation();
            cout << "Neural Network Output: " << out_layer.outputs << endl;
            cout << "Actual Output: " << y_training.row(i) << endl;
            // Find the cost function result
            out_layer.costFunction(y_training.row(i));
            cout << "Cost Function Output: " << endl << out_layer.costResult << endl;
            
            backPropagate(out_layer, hidden_layer, y_training.row(i), x_training.row(i)); 
            /*cout << "New Weights for Hidden are: " << endl;
            cout << hidden_layer.weights << endl;
            cout << "New Weights for Output are: " << endl;
            cout << out_layer.weights << endl;*/ 
        //}
    //}
    /*hidden_layer1.weightedSum(in);
    hidden_layer1.ReLUactivation();
    cout << "Activated hidden layer output: " << endl;
    readData(hidden_layer1.outputs);*/
    
    //cout << "Weights of output layer are: " << endl;
    //readData(out_layer.weights);
    /*out_layer.weightedSum(hidden_layer1.outputs);
    out_layer.Sigactivation();
    cout << "Activated output layer output: " << endl;
    readData(out_layer.outputs);
    MatrixXd out(1, 1);
    out << 0;
    out_layer.costFunction(out);
    cout << "Cost Function Output: " << endl;
    readData(out_layer.costResult);
    cout << "Hidden Layer Old Weights: " << endl;
    readData(hidden_layer1.weights);
    cout << "Output Layer Old Weights: " << endl;
    readData(out_layer.weights);
    backPropagate(out_layer, hidden_layer1, out, in);
     cout << "Hidden Layer New Weights: " << endl;
    readData(hidden_layer1.weights);
    cout << "Output Layer New Weights: " << endl;
    readData(out_layer.weights);*/

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