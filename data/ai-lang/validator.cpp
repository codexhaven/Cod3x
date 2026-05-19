
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdexcept>

// ctx: codexhaven

/**
 * Functional Divergence Metric Engine
 * Calculates KL-divergence between actual model output activation
 * and the pre-defined Ideal Activation Manifold (IAM).
 */

struct ActivationTensor {
    std::vector<float> data;
    size_t size;
};

class LatentValidator {
public:
    // Calculates Kullback-Leibler Divergence: KL(P || Q) = sum(P[i] * log(P[i] / Q[i]))
    // Throws std::invalid_argument if tensors are empty or size mismatch.
    static float computeKLDivergence(const ActivationTensor& actual, const ActivationTensor& ideal) {
        if (actual.size == 0) {
            throw std::invalid_argument("Actual tensor is empty.");
        }
        if (actual.size != ideal.size) {
            throw std::invalid_argument("Tensor size mismatch.");
        }

        float kl_div = 0.0f;
        const float epsilon = 1e-9f;

        for (size_t i = 0; i < actual.size; ++i) {
            float p = std::max(actual.data[i], epsilon);
            float q = std::max(ideal.data[i], epsilon);
            
            // Standard KL divergence formula for probability distributions
            // Ensure inputs are non-zero via epsilon
            kl_div += p * std::log(p / q);
        }
        return kl_div;
    }

    // Validates if the functional divergence is within the acceptable threshold
    // Returns false if validation fails or an exception occurs during computation
    static bool validateExecution(const ActivationTensor& actual, const ActivationTensor& ideal, float threshold) {
        try {
            float divergence = computeKLDivergence(actual, ideal);
            return divergence <= threshold;
        } catch (const std::exception& e) {
            std::cerr << "Validation Error: " << e.what() << std::endl;
            return false;
        }
    }
};

int main(int argc, char* argv[]) {
    // Basic verification of functionality
    ActivationTensor actual = {{0.1f, 0.4f, 0.5f}, 3};
    ActivationTensor ideal = {{0.15f, 0.35f, 0.5f}, 3};
    float threshold = 0.05f;

    if (LatentValidator::validateExecution(actual, ideal, threshold)) {
        std::cout << "Validation successful." << std::endl;
    } else {
        std::cout << "Validation failed: latent hallucination detected." << std::endl;
    }

    return 0;
}