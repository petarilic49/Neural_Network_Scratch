#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "eigen-3.4-rc1/Eigen/Dense" // Library for matrix algebra
#include <time.h> // Used to create randomly generated weights between 0 and 1
#include <sstream> //Used to convert the string values into integer values
#include <fstream> //Input/output stream class to operate on files


using namespace std;
using Eigen::MatrixXd;

// Create a class that will stores a dense layer
class neural_layer{
    public:
    // Initialize the weights, biases, and outputs for each dense layer
    MatrixXd weights, biases, outputs;
    // Initialize the result of the cost function. The hidden layer will have a costResult of 0 while the output layer will have a non zero costResult since were comparing its output to the actual
    double costResult;
    // Initialize the number of neurons per layer
    int neurons;
        neural_layer(int num_inputs, int num_neurons){ //Constructor
            // Need to randomly generate weights for each input to each neuron
            neurons = num_neurons;
            // Initialize the size/dimension of the weight, bias, and output matrices and set the values to 0 
            weights.resize(num_inputs, num_neurons);
            weights.setZero();
            biases.resize(1, num_neurons);
            biases.setZero();
            outputs.resize(1, neurons);
            outputs.setZero();
            // Initialize the costResult for the cost function
            costResult = 0;
            // Randomly populate the weight matrix to a value between -1 and 1
            for(int i = 0; i<weights.rows(); i++){
                for(int j = 0; j<weights.cols(); j++){
                    weights(i, j) = (double)rand()/RAND_MAX - 0.35; // Added the -0.35 to offset the weights and make sure we have negative weights
                }
            }
            return;
        }
        // Performs forward propogation on the layer, ie generates the weighted sum output by input * weights + biases
        void weightedSum(MatrixXd inputs){
            // Eigen library allows you to perform matrix multiplication with just * operand
            outputs = inputs * weights + biases;
            return;
        }
        // Performs the ReLU activation function on the weighted sum output, this function is used only with the hidden layer
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
        // Performs the sigmoid activation function on the weighted sum output, this function is used only with the output layer to get a probability output between 0 and 1
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
        // Since this is a binary classification problem it is best to use a binary cross entropy cost function to for Gradient Decent Learning  
        // Cost Function = -ylog(y_hat) - (1-y)log(1-y_hat)
        void costFunction (MatrixXd actuals){ // Assuming that we are training the model with a batch size of just 1 since the dataset is so small 
            double hold = 0;
            if(actuals(0, 0) == 1){
                    hold = (-1.0 * log(outputs(0,0)));
            }
            else if(actuals(0,0) == 0){
                    hold = (-1.0 * log(1 - outputs(0,0)));
            }
            if(hold == -0){// Sometimes if the log answer is 0 the output becomes -0, this if statement eliminates the negative to keep the data clean throughout the program
                hold = 0;
            }
            costResult = hold;
        }
};

// Function purpose is to perform backpropagation through the neural network by calculating the partial derivatives of the cost function with respect to each weight and bias in the network (ie perform gradient descent and apply the change to the weight/bias)
void backPropagate(neural_layer &out, neural_layer &hidden, MatrixXd actuals, MatrixXd inputs){ 
    // Initialize a column vector that will hold the gradients for each weight and set to zero
    MatrixXd C_p_w((out.weights.rows() * out.weights.cols()) + (hidden.weights.rows() * hidden.weights.cols()), 1); // Gradient vector (ie holds the gradient with respect to each weight)
    C_p_w.setZero();
    // Initialize a column vector that will hold the gradients for each bias and set to zero
    MatrixXd C_p_b((out.neurons + hidden.neurons), 1);
    C_p_b.setZero();
    // Initialize the partial derivitive of the cost function with respect to the output
    double C_p_y = 0; // Only one value because there is only one output from the neural network

    // Initialize count to zero, this variable is used to keep track of the row index of the C_p_y variable within the for loops below
    int count = 0;

    // Calculate the partial derivative of the cost function wrt output, derivation shows its just predicted output - actual output
    C_p_y = (out.outputs(0,0) - actuals(0,0)); 
    // Double for loop, first for loop pertains to the weights connecting from the hidden layer to the output layer (or node just because theres only one output)
    // The second for loop loops through the weights corresponding to the hidden layer neurons. Specifically the outer for loop will update the weight of the first neuron of hidden layer
    // to the output node, then the inner for loop will update all the weights from the first neuron to the inputs
    // Therefore you can think of the for loops as updating all weights attached to each hidden layer neuron from top to bottom
    for(int i = 0; i<hidden.outputs.cols(); i++){ // .cols() for the 15 outputs corresponding to each weight into the output neuron
        C_p_w(count, 0) = C_p_y * hidden.outputs(0,i);
        count++;
        for(int j = 0; j<hidden.weights.rows(); j++){ // .ros() for the 13 inputs the network is fed
            // Initialize the derivative of the ReLU activation function with respect to the weighted sum, this is just either 1 or 0 depending on if the weighted sum is greater than 0 or not
            double der = 0;
            if(hidden.outputs(0, i)>0){
                der = 1;
            }else{
                der = 0;
            }
            C_p_w(count, 0) = C_p_y * out.weights(i, 0) * der * inputs(0,j);
            if(C_p_w(count, 0) == -0){ // Again this if statement just eliminates any -0 values and keeps the data flow clean
                C_p_w(count, 0) = 0;
            }
            count++;
            
        }
    }

    // For loop that will get the gradients for the biases
    // Only need one for loop since the derivation is much simpler, the column vector will hold the bias gradients for each neuron starting from output to the bottom of the hidden layer neuron
    for(int i = 0; i<C_p_b.rows(); i++){
        if(i == 0){ // The output neurons bias gradient will just be equal to predicted output  - actual output
            C_p_b(i, 0) = C_p_y;
        }
        else{ // Hidden layer neuron bias gradients
            double der = 0;
            if(hidden.outputs(0,i-1) > 0){
                der = 1;
            }
            else{
                der = 0;
            }
            C_p_b(i,0) = C_p_y * der;
            if(C_p_b(i,0) == -0){
                C_p_b(i,0) = 0;
            }
        }
    }
 
   
    // Double for loop that will update all the weights in the network based on the calculated gradient per weight and a preset constant learning rate
    // Initialize a learning rate and set to 0.005
    double learn_rate = 0.005;
    // Once again reset the count variable which will keep track of the index for each weight in the C_p_w column vector
    count = 0;
    for(int i = 0; i<out.weights.rows(); i++){ 
        out.weights(i, 0) = out.weights(i, 0) - learn_rate*C_p_w(count, 0);
        count++;
        for(int j = 0; j<hidden.weights.rows(); j++){
            hidden.weights(j, i) = hidden.weights(j, i) - learn_rate*C_p_w(count, 0);
            count++;
        }
    }
    // For loop that will update the biases in the network based on the calculated gradient per bias and the preset constant learning rate
    for(int i = 0; i<C_p_b.rows(); i++){
        if(i == 0){
            out.biases(0,0) = out.biases(0,0) - learn_rate*C_p_b(i, 0);
        }
        else{
            hidden.biases(0,i-1) = hidden.biases(0,i-1) - learn_rate*C_p_b(i, 0);
        }
        
    }
    
    return;
}

// Function that will take the .csv file with the dataset and convert the data into a matrix 
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

    // Second loop to populate the matrix with the data values
    mat.resize(num_rows, num_cols);
    dataFile.open(fileName); //Reopen the csv file from the top
    num_rows = 0;
    while(getline(dataFile, line)){
        num_cols = 0;
        stringstream linestream(line);
        // While loop which populates the data from the .csv file into the matrix but also converts the values from strings to doubles to allow us to perform math operations
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

// Function that will remove a row from the matrix dataset
void removeRow(MatrixXd &mat, int rowtoRemove){
    // Initialize the new row number and column number
    int rowNum = mat.rows() - 1;
    int colNum = mat.cols();

    if(rowtoRemove<rowNum){
        // Use .block() to set the new portion of the matrix to the portion under the rowtoremove variable
        mat.block(rowtoRemove, 0, rowNum - rowtoRemove, colNum) = mat.block(rowtoRemove+1, 0, rowNum - rowtoRemove, colNum);
    }
    // Resize the new matrix
    mat.conservativeResize(rowNum, colNum);
    return;
}

// Output the matrix in the terminal (mainly used for debugging purposes)
void readData(MatrixXd &d){
    cout << d << endl;
    return;
}

// Function that will split the dataset into training and testing
// Function works by randomly reorganize the dataset and then split the training set to 70% and the testinb to 30%
void splitData(MatrixXd &td, MatrixXd &xtrain, MatrixXd &ytrain, MatrixXd &xtest, MatrixXd &ytest){
    int numRows = td.rows();
    int numCols = td.cols();
    int trainRows = round(numRows * 0.7);

    // Create a matrix that is randomly sorted with the total data set
    MatrixXd random(numRows, numCols);
    // Keep track which rows have been populated (using a vector because it can dynamically change size)
    vector<int> rowCount;

    int rowindex = 0;
    // Loop through the dataset and randomly pick a row index
    // The second for loop is used to check if the randomly generated row index has already been taken by checking the rowCount vector, if it has then pick another random row index and restart the for loop to check 
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
        // Push the unchecked row index to the rowCount vector and update the random matrix with indexed row from the dataset
        rowCount.push_back(rowindex);
        random.row(i) = td.row(rowindex); 
    }

    // Split the training and testing data with the random matrix
    xtrain = random.block(0, 0, trainRows, numCols - 1);
    ytrain = random.block(0, numCols-1, trainRows, 1);
    xtest = random.block(trainRows, 0, numRows - trainRows, numCols - 1);
    ytest = random.block(trainRows, numCols - 1, numRows - trainRows, 1);

    return;
}

// Function is used to normalize the data in the xtrain and xtest
void normalizeData(MatrixXd &m, int rows, int cols){
    // Initialize the min as very high number and max as zero
    int min = 10000;
    int max = 0;
    int colHold = 0;

    // Loop through the the data set and determine the maximum and minum value per column
    while(colHold < cols){
        // For loop determines the minimum and maximum value in the column of the matrix
        for(int i = 0; i<rows; i++){
            if(m(i, colHold) < min){
                min = m(i, colHold);
            }
            else if(m(i, colHold) > max){
                max = m(i, colHold);
            }
        }
        // For loop which iterates through each value in the column of the matrix and normalizes the data entries using the determined max and min values
        for (int i  = 0; i<rows; i++){
            double val;
            val = (m(i, colHold) - min) / (max - min);
            m(i, colHold) = val;
        }
        // Reset the max and min value and repeat for next column of the matrix 
        colHold++;
        max = 0;
        min = 10000;
    }
    return;
}

// Function that writes to a .csv file (used for analyzing the epoch vs loss as well as a visual to how good the network is)
bool writetoCSV(const string filename, const string epoch, const string loss){
    ofstream lossfile;
    lossfile.open(filename, ios_base::app); // The ios_base::app make sures that the new data is appended into the .csv file and does not overwrite what is there
    lossfile << epoch << "," << loss << endl;
    lossfile.close();

    return true;
}

// Function that performs forward propagation through the trained neural network and calculates the accruacy of the network (roughly)
// For simplicity accuracy in this case will be equal to number of correct predictions / number of total predictions
void predict(MatrixXd x_test, MatrixXd y_test, MatrixXd x_train, MatrixXd y_train, neural_layer &hidden, neural_layer &out){ 
    // Initialize the counter that will store the number of correct predictions as well as the accuracy 
    double correct = 0, accuracy = 0;
    // Strings that will hold the prediction and actual value which will be sent to the writetoCSV function (fstream only allows you to write strings not doubles)
    string s_pred, s_actual;
    cout << "Training Data Accuracy" << endl;
    // For loop that will perform forward propagation through the neural network using the training data. This is done as the values will be stored in an excel file which will be analyzed
    for(int i = 0; i<x_train.rows(); i++){
        hidden.weightedSum(x_train.row(i));
        hidden.ReLUactivation();
        out.weightedSum(hidden.outputs);
        out.Sigactivation();
        // Store the values of the predicted value and actual value output and send to writetoCSV function which will update an excel file with all the outputs
        s_pred = to_string(out.outputs(0,0));
        s_actual = to_string(y_train(i,0));
        bool updatePrediction = writetoCSV("Predictions vs Actual.csv", s_pred, s_actual);
        // The below if statement is treated as a cutoff value to determine which predicted outputs were correct and which were not
        if(out.outputs(0,0) > 0.9 && y_train(i,0) == 1){
            correct++;
        }
        else if(out.outputs(0,0) < 0.1 && y_train(i,0) == 0){
            correct++;
        }
    }
    // Caluclate the training accuracy with the determined epoch size
    accuracy = 100*(correct/x_train.rows());
    cout << "Training Accuracy is: " << accuracy << "%" << endl;
    correct = 0;
    bool updatePrediction = writetoCSV("Predictions vs Actual.csv", "Testing Predictions", "Testing Actuals");
    // For loop that will perform forward propagation through the neural network using the testing data
    for(int i = 0; i<x_test.rows(); i++){
        hidden.weightedSum(x_test.row(i));
        hidden.ReLUactivation();
        out.weightedSum(hidden.outputs);
        out.Sigactivation();
        s_pred = to_string(out.outputs(0,0));
        s_actual = to_string(y_test(i,0));
        bool updatePrediction = writetoCSV("Predictions vs Actual.csv", s_pred, s_actual);
        if(out.outputs(0,0) > 0.9 && y_test(i,0) == 1){
            correct++;
        }
        else if(out.outputs(0,0) < 0.1 && y_test(i,0) == 0){
            correct++;
        }
    }
    accuracy = 100*(correct/x_test.rows());
    cout << "Testing Accuracy is: " << accuracy << "%" << endl;
    return;
}

int main()
{
    // Initialize a matrix that will hold the whole dataset
    MatrixXd data;
    // Call function to read the .csv file and update the data matrix 
    readDatatoMat("heart.csv", data);
    // Remove the first row of the matrix since that row holds the title of the dataset (ie in this case they were all 0)
    removeRow(data, 0);

    // Initialize matrices which will hold the testing and training data
    MatrixXd x_training, y_training, x_testing, y_testing;
    // Split the test and trianing data 
    splitData(data, x_training, y_training, x_testing, y_testing);
    // Normalize just the x data since those are the inputs
    normalizeData(x_training, x_training.rows(), x_training.cols());
    normalizeData(x_testing, x_testing.rows(), x_testing.cols());

    //Create the hidden layer that has 15 neurons and the output layer which has only 1 neuron since its a binary classification problem
    neural_layer hidden_layer(x_training.cols(), 15);
    neural_layer out_layer(15, 1);
    
    // Initialize a variable that will hold the accumulated sum of the cost function output, this will be used to determine if the network is learning
    double loss = 0;
    // Initialize strings that will hold the epoch number and loss which will be used to update a .csv file 
    string s_epoch, s_loss;
    cout << "Training the network" << endl;
    // Need a double for loop, outer for the epoch, inner to go through the whole training dataset
    // Epoch is set to be at 1500 through trial and error
    for(int epoch = 0; epoch<1500; epoch++){
        for(int i = 0; i<x_training.rows(); i++){
            // Forward propagate the neural network
            hidden_layer.weightedSum(x_training.row(i));
            hidden_layer.ReLUactivation();
            out_layer.weightedSum(hidden_layer.outputs);
            out_layer.Sigactivation();
            // Find the cost function result
            out_layer.costFunction(y_training.row(i));
            // Add the new cost function result to the pre existing cost function loss
            loss = loss + out_layer.costResult;
            backPropagate(out_layer, hidden_layer, y_training.row(i), x_training.row(i)); 
        }
        cout << "Total cost for epoch " << epoch << " is " << loss << endl;
        // Update a .csv with the current epoch number and its corresponding accumulated loss which will be used to determine which epoch to choose without overfitting the network
        s_epoch = to_string(epoch);
        s_loss = to_string(loss);
        bool updateLoss = writetoCSV("Epoch vs Loss.csv", s_epoch, s_loss);
        loss = 0;
    }
    cout << "Predicting heart failures" << endl;
    // Create the header column of the predictions vs actual .csv file
    ofstream outputFile;
    outputFile.open("Predictions vs Actual.csv", ios_base::app); // The ios_base::app make sures that the new data is appended into the .csv file and does not overwrite what is there
    outputFile << "Predicted Training, Actual Training" << endl;
    outputFile.close();
    // Take the current weights and biases of the network and predict the future values with the testing data
    predict(x_testing, y_testing, x_training, y_training, hidden_layer, out_layer);
    
    return 0;
}