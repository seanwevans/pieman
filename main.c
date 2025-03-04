#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RANDOM_SEED 1

#define INPUT_SIZE_ORIGINAL 784
#define HIDDEN_LAYERS 28
#define HIDDEN_SIZE_ORIGINAL 28
#define OUTPUT_SIZE_ORIGINAL 10

#define INPUT_SIZE ((INPUT_SIZE_ORIGINAL + 3) & ~3)   // Pad to multiple of 4
#define HIDDEN_SIZE ((HIDDEN_SIZE_ORIGINAL + 3) & ~3) // Pad to multiple of 4
#define OUTPUT_SIZE ((OUTPUT_SIZE_ORIGINAL + 3) & ~3) // Pad to multiple of 4

#define LEARNING_RATE 1e-2
#define SAMPLE_SIZE 1
#define MAX_EPOCHS 1e9
#define MAX_ACCEPTABLE_LOSS 1e-5

#define REPORT_FREQUENCY 10000

typedef struct {
  double *weights;
  double *outputs;
  double *biases;
  double *deltas;
} Layer;

Layer *hidden_layers;
double *input;
double *output_weights;
double *output_biases;
double *output_layer;
double *output_deltas;
double *target;
double *real_target;

void *aligned_malloc(size_t size) {
  void *ptr = aligned_alloc(32, size);
  if (!ptr) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double sigmoid_derivative(double x) { return x * (1.0 - x); }

double horizontal_sum(__m256d v) {
  __m128d high = _mm256_extractf128_pd(v, 1);
  __m128d low = _mm256_castpd256_pd128(v);
  __m128d sum = _mm_add_pd(high, low);
  __m128d shuffle = _mm_shuffle_pd(sum, sum, 1);
  __m128d result = _mm_add_pd(sum, shuffle);
  return _mm_cvtsd_f64(result);
}

void initialize_network() {
  printf("👨‍🎓 %d params\n", HIDDEN_LAYERS * HIDDEN_SIZE_ORIGINAL);
  hidden_layers = (Layer *)malloc(HIDDEN_LAYERS * sizeof(Layer));
  if (!hidden_layers) {
    fprintf(stderr, "Memory allocation failed for hidden layers!\n");
    exit(EXIT_FAILURE);
  }

  input = (double *)aligned_malloc(INPUT_SIZE * sizeof(double));
  output_layer = (double *)aligned_malloc(OUTPUT_SIZE * sizeof(double));
  output_deltas = (double *)aligned_malloc(OUTPUT_SIZE * sizeof(double));
  output_biases = (double *)aligned_malloc(OUTPUT_SIZE * sizeof(double));
  target = (double *)aligned_malloc(OUTPUT_SIZE * sizeof(double));
  real_target = (double *)aligned_malloc(OUTPUT_SIZE_ORIGINAL * sizeof(double));

  memset(input, 0, INPUT_SIZE * sizeof(double));
  memset(target, 0, OUTPUT_SIZE * sizeof(double));

  for (int i = 0; i < HIDDEN_LAYERS; i++) {
    size_t input_dim = (i == 0) ? INPUT_SIZE : HIDDEN_SIZE;
    size_t weight_size = HIDDEN_SIZE * input_dim;

    hidden_layers[i].weights =
        (double *)aligned_malloc(weight_size * sizeof(double));
    hidden_layers[i].outputs =
        (double *)aligned_malloc(HIDDEN_SIZE * sizeof(double));
    hidden_layers[i].biases =
        (double *)aligned_malloc(HIDDEN_SIZE * sizeof(double));
    hidden_layers[i].deltas =
        (double *)aligned_malloc(HIDDEN_SIZE * sizeof(double));

    for (int j = 0; j < HIDDEN_SIZE; j++) {
      hidden_layers[i].biases[j] = (rand() / (double)RAND_MAX) - 0.5;

      for (unsigned k = 0; k < input_dim; k++) {
        hidden_layers[i].weights[j * input_dim + k] =
            (rand() / (double)RAND_MAX) - 0.5;
      }
    }

    memset(hidden_layers[i].outputs, 0, HIDDEN_SIZE * sizeof(double));
    memset(hidden_layers[i].deltas, 0, HIDDEN_SIZE * sizeof(double));
  }

  output_weights =
      (double *)aligned_malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    output_biases[i] = (rand() / (double)RAND_MAX) - 0.5;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
      output_weights[i * HIDDEN_SIZE + j] = (rand() / (double)RAND_MAX) - 0.5;
    }
  }
}

void forward() {
  // First hidden layer
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    __m256d sum_vec = _mm256_set1_pd(hidden_layers[0].biases[j]);

    for (int k = 0; k < INPUT_SIZE; k += 4) {
      __m256d input_vec = _mm256_load_pd(&input[k]);
      __m256d weight_vec =
          _mm256_load_pd(&hidden_layers[0].weights[j * INPUT_SIZE + k]);
      sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
    }

    hidden_layers[0].outputs[j] = sigmoid(horizontal_sum(sum_vec));
  }

  // Remaining hidden layers
  for (int i = 1; i < HIDDEN_LAYERS; i++) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      __m256d sum_vec = _mm256_set1_pd(hidden_layers[i].biases[j]);

      for (int k = 0; k < HIDDEN_SIZE; k += 4) {
        __m256d input_vec = _mm256_load_pd(&hidden_layers[i - 1].outputs[k]);
        __m256d weight_vec =
            _mm256_load_pd(&hidden_layers[i].weights[j * HIDDEN_SIZE + k]);
        sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
      }

      hidden_layers[i].outputs[j] = sigmoid(horizontal_sum(sum_vec));
    }
  }

  // Output layer
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    __m256d sum_vec = _mm256_set1_pd(output_biases[i]);

    for (int j = 0; j < HIDDEN_SIZE; j += 4) {
      __m256d input_vec =
          _mm256_load_pd(&hidden_layers[HIDDEN_LAYERS - 1].outputs[j]);
      __m256d weight_vec = _mm256_load_pd(&output_weights[i * HIDDEN_SIZE + j]);
      sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
    }

    output_layer[i] = sigmoid(horizontal_sum(sum_vec));
  }
}

double loss() {
  double total_loss = 0.0;

  for (int i = 0; i < OUTPUT_SIZE; i += 4) {
    __m256d target_vec = _mm256_load_pd(&target[i]);
    __m256d output_vec = _mm256_load_pd(&output_layer[i]);

    // Calculate error
    __m256d error_vec = _mm256_sub_pd(target_vec, output_vec);

    // Calculate sigmoid derivative
    __m256d one_vec = _mm256_set1_pd(1.0);
    __m256d deriv_vec =
        _mm256_mul_pd(output_vec, _mm256_sub_pd(one_vec, output_vec));

    // Calculate deltas
    __m256d delta_vec = _mm256_mul_pd(error_vec, deriv_vec);
    _mm256_store_pd(&output_deltas[i], delta_vec);

    // Calculate squared error for loss
    __m256d squared_error = _mm256_mul_pd(error_vec, error_vec);
    total_loss += horizontal_sum(squared_error);
  }

  return total_loss / OUTPUT_SIZE_ORIGINAL;
}

void backward() {
  for (int i = HIDDEN_LAYERS - 1; i >= 0; i--) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      __m256d error_vec = _mm256_setzero_pd();

      if (i == HIDDEN_LAYERS - 1) {
        for (int k = 0; k < OUTPUT_SIZE; k += 4) {
          __m256d delta_vec = _mm256_load_pd(&output_deltas[k]);
          __m256d weight_vec =
              _mm256_set_pd(output_weights[(k + 3) * HIDDEN_SIZE + j],
                            output_weights[(k + 2) * HIDDEN_SIZE + j],
                            output_weights[(k + 1) * HIDDEN_SIZE + j],
                            output_weights[k * HIDDEN_SIZE + j]);
          error_vec = _mm256_fmadd_pd(delta_vec, weight_vec, error_vec);
        }
      } else {
        for (int k = 0; k < HIDDEN_SIZE; k += 4) {
          __m256d delta_vec = _mm256_load_pd(&hidden_layers[i + 1].deltas[k]);
          __m256d weight_vec = _mm256_set_pd(
              hidden_layers[i + 1].weights[(k + 3) * HIDDEN_SIZE + j],
              hidden_layers[i + 1].weights[(k + 2) * HIDDEN_SIZE + j],
              hidden_layers[i + 1].weights[(k + 1) * HIDDEN_SIZE + j],
              hidden_layers[i + 1].weights[k * HIDDEN_SIZE + j]);
          error_vec = _mm256_fmadd_pd(delta_vec, weight_vec, error_vec);
        }
      }
      double error = horizontal_sum(error_vec);
      hidden_layers[i].deltas[j] =
          error * sigmoid_derivative(hidden_layers[i].outputs[j]);
    }
  }

  for (int i = 0; i < OUTPUT_SIZE; i++) {
    __m256d delta_vec = _mm256_set1_pd(output_deltas[i]);
    __m256d lr_vec = _mm256_set1_pd(LEARNING_RATE);

    for (int j = 0; j < HIDDEN_SIZE; j += 4) {
      __m256d hidden_output_vec =
          _mm256_load_pd(&hidden_layers[HIDDEN_LAYERS - 1].outputs[j]);
      __m256d weight_update =
          _mm256_mul_pd(_mm256_mul_pd(lr_vec, delta_vec), hidden_output_vec);

      __m256d current_weights =
          _mm256_load_pd(&output_weights[i * HIDDEN_SIZE + j]);
      __m256d new_weights = _mm256_add_pd(current_weights, weight_update);
      _mm256_store_pd(&output_weights[i * HIDDEN_SIZE + j], new_weights);
    }

    output_biases[i] += LEARNING_RATE * output_deltas[i];
  }

  for (int i = HIDDEN_LAYERS - 1; i >= 0; i--) {
    size_t prev_size = (i > 0) ? HIDDEN_SIZE : INPUT_SIZE;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
      __m256d delta_vec = _mm256_set1_pd(hidden_layers[i].deltas[j]);
      __m256d lr_vec = _mm256_set1_pd(LEARNING_RATE);

      for (unsigned k = 0; k < prev_size; k += 4) {
        __m256d prev_output_vec;
        if (i > 0) {
          prev_output_vec = _mm256_load_pd(&hidden_layers[i - 1].outputs[k]);
        } else {
          prev_output_vec = _mm256_load_pd(&input[k]);
        }

        __m256d weight_update =
            _mm256_mul_pd(_mm256_mul_pd(lr_vec, delta_vec), prev_output_vec);

        __m256d current_weights =
            _mm256_load_pd(&hidden_layers[i].weights[j * prev_size + k]);
        __m256d new_weights = _mm256_add_pd(current_weights, weight_update);
        _mm256_store_pd(&hidden_layers[i].weights[j * prev_size + k],
                        new_weights);
      }

      hidden_layers[i].biases[j] += LEARNING_RATE * hidden_layers[i].deltas[j];
    }
  }
}

void cleanup() {
  for (int i = 0; i < HIDDEN_LAYERS; i++) {
    free(hidden_layers[i].weights);
    free(hidden_layers[i].outputs);
    free(hidden_layers[i].biases);
    free(hidden_layers[i].deltas);
  }

  free(hidden_layers);
  free(input);
  free(output_weights);
  free(output_biases);
  free(output_layer);
  free(output_deltas);
  free(target);
  free(real_target);
}

int save_model(const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
    return 0;
  }

  for (int i = 0; i < HIDDEN_LAYERS; i++) {
    size_t input_dim = (i == 0) ? INPUT_SIZE : HIDDEN_SIZE;
    size_t weight_size = HIDDEN_SIZE * input_dim;

    if (fwrite(hidden_layers[i].weights, sizeof(double), weight_size, file) !=
        weight_size) {
      fprintf(stderr, "Error writing hidden layer %d weights\n", i);
      fclose(file);
      return 0;
    }

    if (fwrite(hidden_layers[i].biases, sizeof(double), HIDDEN_SIZE, file) !=
        HIDDEN_SIZE) {
      fprintf(stderr, "Error writing hidden layer %d biases\n", i);
      fclose(file);
      return 0;
    }
  }

  if (fwrite(output_weights, sizeof(double), OUTPUT_SIZE * HIDDEN_SIZE, file) !=
      OUTPUT_SIZE * HIDDEN_SIZE) {
    fprintf(stderr, "Error writing output weights\n");
    fclose(file);
    return 0;
  }

  if (fwrite(output_biases, sizeof(double), OUTPUT_SIZE, file) != OUTPUT_SIZE) {
    fprintf(stderr, "Error writing output biases\n");
    fclose(file);
    return 0;
  }

  fclose(file);
  printf("💾 %s\n", filename);
  return 1;
}

int load_model(const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Could not open model file %s\n", filename);
    return 0;
  }

  for (int i = 0; i < HIDDEN_LAYERS; i++) {
    size_t input_dim = (i == 0) ? INPUT_SIZE : HIDDEN_SIZE;
    size_t weight_size = HIDDEN_SIZE * input_dim;

    if (fread(hidden_layers[i].weights, sizeof(double), weight_size, file) !=
        weight_size) {
      fprintf(stderr, "Error reading hidden layer %d weights\n", i);
      fclose(file);
      return 0;
    }
    if (fread(hidden_layers[i].biases, sizeof(double), HIDDEN_SIZE, file) !=
        HIDDEN_SIZE) {
      fprintf(stderr, "Error reading hidden layer %d biases\n", i);
      fclose(file);
      return 0;
    }
  }

  if (fread(output_weights, sizeof(double), OUTPUT_SIZE * HIDDEN_SIZE, file) !=
      OUTPUT_SIZE * HIDDEN_SIZE) {
    fprintf(stderr, "Error reading output weights\n");
    fclose(file);
    return 0;
  }

  if (fread(output_biases, sizeof(double), OUTPUT_SIZE, file) != OUTPUT_SIZE) {
    fprintf(stderr, "Error reading output biases\n");
    fclose(file);
    return 0;
  }

  fclose(file);
  return 1;
}

void load_data() {
  for (unsigned i = 0; i < INPUT_SIZE_ORIGINAL; i++) {
    input[i] = rand() / (double)RAND_MAX;
  }
  // Pad remaining input elements with zeros
  for (unsigned i = INPUT_SIZE_ORIGINAL; i < INPUT_SIZE; i++) {
    input[i] = 0.0;
  }

  for (unsigned i = 0; i < OUTPUT_SIZE_ORIGINAL; i++) {
    real_target[i] = rand() / (double)RAND_MAX;
    target[i] = real_target[i];
  }
  // Pad remaining target elements with zeros
  for (unsigned i = OUTPUT_SIZE_ORIGINAL; i < OUTPUT_SIZE; i++) {
    target[i] = 0.0;
  }
}

void train() {
  int epoch = 0;
  double _loss = 100.0;
  puts("┏━━━━━━━━━━━━━━━━━━━━━━┓");
  puts("┃   Epoch  Loss        ┃");
  puts("┣━━━━━━━━━━━━━━━━━━━━━━┫");

  clock_t start = clock();
  while (_loss > MAX_ACCEPTABLE_LOSS && epoch < MAX_EPOCHS) {
    epoch += 1;
    forward();
    _loss = loss();
    backward();

    if (epoch % REPORT_FREQUENCY == 0 || epoch == 1) {
      printf("┃ %7d  %.9f ┃\n", epoch, _loss);
    }
  }

  printf("┃ %7d  %.9f ┃\n"
         "┗━━━━━━━━━━━━━━━━━━━━━━┛\n"
         "🏋️ %f seconds\n",
         epoch, _loss, (double)(clock() - start) / CLOCKS_PER_SEC);

  /*printf(
      "┃ %7d  %.9f ┃\n"
      "┗━━━━━━━━━━━━━━━━━━━━━━┛\n",
      epoch, _loss
  );*/
}

/*
#define RANDOM_SEED 1
#define INPUT_SIZE_ORIGINAL 784
#define HIDDEN_LAYERS 28
#define HIDDEN_SIZE_ORIGINAL 28
#define OUTPUT_SIZE_ORIGINAL 10
#define INPUT_SIZE  (( INPUT_SIZE_ORIGINAL + 3) & ~3) // Pad to multiple of 4
#define HIDDEN_SIZE ((HIDDEN_SIZE_ORIGINAL + 3) & ~3) // Pad to multiple of 4
#define OUTPUT_SIZE ((OUTPUT_SIZE_ORIGINAL + 3) & ~3) // Pad to multiple of 4
#define LEARNING_RATE 1e-2
#define SAMPLE_SIZE 1
#define MAX_EPOCHS (1<<31)
#define MAX_ACCEPTABLE_LOSS 1e-5
#define REPORT_FREQUENCY 10000

    👨‍🎓 784 params
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃   Epoch  Loss        ┃
    ┣━━━━━━━━━━━━━━━━━━━━━━┫
    ┃       1  0.306073244 ┃
    ┃   10000  0.000067958 ┃
    ┃   20000  0.000032532 ┃
    ┃   30000  0.000021231 ┃
    ┃   40000  0.000015706 ┃
    ┃   50000  0.000012440 ┃
    ┃   60000  0.000010286 ┃
    ┃   61650  0.000010000 ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━┛
    🏋️ 1.160054 seconds
    💾 model.bin
*/

int main() {
  srand(RANDOM_SEED);

  initialize_network();
  load_data();

  train();
  save_model("model.bin");

  cleanup();
  return 0;
}
