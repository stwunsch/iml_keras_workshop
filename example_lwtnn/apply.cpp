#include <iostream>
#include <string>
#include <fstream>
#include "stdlib.h"

#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"

int main(int argc, char* argv[]){
    if(argc!=5){
        std::cout << "Usage: apply sepal_length sepal_width petal_length petal_width" << std::endl;
        return 1;
    }

    // Read lwtnn JSON config
    std::string config_filename("lwtnn.json");
    std::ifstream config_file(config_filename);
    auto config = lwt::parse_json(config_file);

    // Set up neural network model from config
    lwt::LightweightNeuralNetwork model(
            config.inputs,
            config.layers,
            config.outputs);

    // Load inputs from argv
    std::map<std::string, double> inputs;
    for(size_t n=0; n < config.inputs.size(); n++)
            inputs[config.inputs.at(n).name] = std::atof(argv[n+1]);

    // Apply model on inputs
    auto outputs = model.compute(inputs);

    // Print computed probabilities for classes
    for(const auto& output: outputs)
            std::cout << output.first << " " << output.second << std::endl;
}
