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

bool writetoCSV(const string filename, const string epoch, const string loss);

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
            if(hold == -0){
                hold = 0;
            }
            costResult = hold;
        }
};

void backPropagate(neural_layer &out, neural_layer &hidden, MatrixXd actuals, MatrixXd inputs, bool check){ // Function is used to backpropagate through the specific layer and update the weights
    // Start with the output layer (ie in out case its just the weights of the output neuron)
    // Three partial derivatives need to be mulitplied (use naming convention as: y_p_x partial derivative of y with respect to x)
    MatrixXd C_p_w((out.weights.rows() * out.weights.cols()) + (hidden.weights.rows() * hidden.weights.cols()), 1); // Gradient vector (ie holds the gradient with respect to each weight)
    C_p_w.setZero();
    double C_p_y = 0; // Only one value because there is only one output from the neural network
    //double y_p_z = 0;
    int count = 0;
    C_p_y = (out.outputs(0,0) - actuals(0,0)); 
    //cout << "C_p_y: " << C_p_y << endl;
    for(int i = 0; i<hidden.outputs.cols(); i++){
        // Find the gradient of the first output 
        C_p_w(count, 0) = C_p_y * hidden.outputs(0,i);
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
            C_p_w(count, 0) = C_p_y * out.weights(i, 0) * der * inputs(0,j);
            if(C_p_w(count, 0) == -0){
                C_p_w(count, 0) = 0;
            }
            if(check){
                cout << "Gradient val is: " << C_p_y << " * " << out.weights(i,0) << " * " << der << " * " << inputs(0,j) << " = " << C_p_w(count, 0) << endl;
            }
            count++;
            
        }
    }
   
    // Chunk that update the weights of the layers
    double learn_rate = 0.005;
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

void predict(MatrixXd x_test, MatrixXd y_test, MatrixXd x_train, MatrixXd y_train, neural_layer &hidden, neural_layer &out){ // Function that will take the neural layers and just do forward propagation to get the outputs and compare to the actual output
    // Accuracy is equal to number of correct predictions / number of total predictions (for simplicity im gonna use a cut off to see if its right or wrong)
    double correct = 0, accuracy = 0;
    string s_pred, s_actual;
    cout << "Training Data Accuracy" << endl;
    for(int i = 0; i<x_train.rows(); i++){
        hidden.weightedSum(x_train.row(i));
        hidden.ReLUactivation();
        out.weightedSum(hidden.outputs);
        out.Sigactivation();
        s_pred = to_string(out.outputs(0,0));
        s_actual = to_string(y_train(i,0));
        bool updatePrediction = writetoCSV("Predictions vs Actual.csv", s_pred, s_actual);
        //cout << "Predicted Value is: " << out.outputs << endl;
        //cout << "Actual Value is: " << y_test(i,0) << endl;
        if(out.outputs(0,0) > 0.9 && y_train(i,0) == 1){
            correct++;
        }
        else if(out.outputs(0,0) < 0.1 && y_train(i,0) == 0){
            correct++;
        }
    }
    accuracy = 100*(correct/x_train.rows());
    cout << "Training Accuracy is: " << accuracy << "%" << endl;
    correct = 0;
    bool updatePrediction = writetoCSV("Predictions vs Actual.csv", "Testing Predictions", "Testing Actuals");
    for(int i = 0; i<x_test.rows(); i++){
        hidden.weightedSum(x_test.row(i));
        hidden.ReLUactivation();
        out.weightedSum(hidden.outputs);
        out.Sigactivation();
        s_pred = to_string(out.outputs(0,0));
        s_actual = to_string(y_test(i,0));
        bool updatePrediction = writetoCSV("Predictions vs Actual.csv", s_pred, s_actual);
        //cout << "Predicted Value is: " << out.outputs << endl;
        //cout << "Actual Value is: " << y_test(i,0) << endl;
        if(out.outputs(0,0) > 0.9 && y_test(i,0) == 1){
            correct++;
        }
        else if(out.outputs(0,0) < 0.1 && y_test(i,0) == 0){
            correct++;
        }
    }
    accuracy = 100*(correct/x_test.rows());
    cout << "Testing Accuracy is: " << accuracy << "%" << endl;
    //cout <<"Accuracy of the model is: " << accuracy << "%" << endl;
    return;
}

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

void readData(MatrixXd &d){
    cout << d << endl;
    return;
}

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

// Function that writes to a .csv file which will show the decrease in loss over time with each iteration of epoch
bool writetoCSV(const string filename, const string epoch, const string loss){
    ofstream lossfile;
    lossfile.open(filename, ios_base::app); // The ios_base::app make sures that the new data is appended into the .csv file and does not overwrite what is there
    lossfile << epoch << "," << loss << endl;
    lossfile.close();

    return true;
}

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
    double loss = 0;
    string s_epoch, s_loss;
    cout << "Training the network" << endl;
    for(int epoch = 0; epoch<1300; epoch++){
        for(int i = 0; i<x_training.rows(); i++){
            // Forward propagate the neural network
            //cout << "Input to hidden layer: " << x_training.row(i) << endl;
            hidden_layer.weightedSum(x_training.row(i));
            hidden_layer.ReLUactivation();
            //cout << "Hidden Layer Output is: " << hidden_layer.outputs << endl;
            //cout << "Input to output layer: " << hidden_layer.outputs << endl;
            out_layer.weightedSum(hidden_layer.outputs);
            //cout << "Output Layer Weights: " << out_layer.weights << endl;
            //cout << "Output Layer Weighted Sum: " << out_layer.outputs << endl;
            out_layer.Sigactivation();
            //cout << "Neural Network Output: " << out_layer.outputs << endl;
            //cout << "Actual Output: " << y_training.row(i) << endl;
            // Find the cost function result
            out_layer.costFunction(y_training.row(i));
            //cout << "Cost Function Output: " << endl << out_layer.costResult << endl;
            loss = loss + out_layer.costResult;
            /*if((epoch == 1871 && i == 208) || (epoch == 1871 && i == 209)){
                //cout<<"Epoch is: " << epoch << " and iteration is: " << i << endl;
                cout << "The weights of hidden layer with epoch " << epoch << " is: " << endl << hidden_layer.weights << endl;
                cout << "The output of the hidden layer is: " << hidden_layer.outputs << endl;
                cout << "The weights of the output layer with epoch " << epoch << " is: " << endl << out_layer.weights << endl;
                cout << "Neural Network Output: " << out_layer.outputs << endl;
                cout << "Actual Output: " << y_training.row(i) << endl;
                cout << "Cost Function Output: " << endl << out_layer.costResult << endl;
                backPropagate(out_layer, hidden_layer, y_training.row(i), x_training.row(i), 1);
            }*/
            /*if(out_layer.outputs(0,0) == 1){
                cout << "The weights of the hidden layer are: " << endl << hidden_layer.weights << endl;
                cout << "The weights of the output layer are: " << endl << out_layer.weights << endl;
            }*/
            backPropagate(out_layer, hidden_layer, y_training.row(i), x_training.row(i), 0); 
            /*cout << "New Weights for Hidden are: " << endl;
            cout << hidden_layer.weights << endl;
            cout << "New Weights for Output are: " << endl;
            cout << out_layer.weights << endl;*/ 
        }
        //avgerror = avgerror / x_training.rows();
        cout << "Total cost for epoch " << epoch << " is " << loss << endl;
        //s_epoch = to_string(epoch);
        //s_loss = to_string(loss);
        //bool updateLoss = writetoCSV("Epoch vs Loss.csv", s_epoch, s_loss);
        loss = 0;
    }
    cout << "Predicting heart failures" << endl;
    // Create the header column of the predictions vs actual .csv file
    ofstream outputFile;
    outputFile.open("Predictions vs Actual.csv", ios_base::app); // The ios_base::app make sures that the new data is appended into the .csv file and does not overwrite what is there
    outputFile << "Predicted Training, Actual Training" << endl;
    outputFile.close();
    predict(x_testing, y_testing, x_training, y_training, hidden_layer, out_layer);
    
    return 0;
}