
#include <math.h>
#include <mpi.h>

#include <iostream>

// Initial coordinates
const double X_0 = -1.0;
const double Y_0 = -1.0;
const double Z_0 = -1.0;

// Dimension size
const double D_X = 2.0;
const double D_Y = 2.0;
const double D_Z = 2.0;

// Grid size
const int N_X = 320;
const int N_Y = 320;
const int N_Z = 320;

// Step size
const double H_X = (D_X / (N_X - 1));
const double H_Y = (D_Y / (N_Y - 1));
const double H_Z = (D_Z / (N_Z - 1));

// Square of step size
const double H_X_2 = (H_X * H_X);
const double H_Y_2 = (H_Y * H_Y);
const double H_Z_2 = (H_Z * H_Z);

// Parameters
const double A = 1.0E5;
const double EPSILON = 1.0E-8;

double phi(double x, double y, double z);
double rho(double x, double y, double z);

int getIndex(int x, int y, int z);
double getX(int i);
double getY(int j);
double getZ(int k);
void setMatrixParts(int *layerHeights, int *offsets, int processSize);
void initLayers(double *prevLayer, double *currentLayer, int layerHeight,
                int offset);
double calcCenter(const double *prevLayer, double *currentLayer,
                  int layerHeight, int offset);
double calcUpBorder(const double *prevLayer, double *currentLayer,
                    const double *upBorderLayer, int layerHeight, int offset,
                    int processRank, int processSize);
double calcDownBorder(const double *prevLayer, double *currentLayer,
                      const double *downBorderLayer, int layerHeight,
                      int offset, int processRank, int processSize);
double calcMaxDiff(const double *func, int layerHeight, int offset);
void changeMaxDiff(const double phi_i, int offset, double &maxDiff, int i,
                   int j, int k, const double *prevLayer, double *currentLayer);

int main(int argc, char **argv) {
  int processRank, processSize;
  double startTime, endTime;
  double maxDiff = 0.0;
  int *layerHeights = NULL;
  int *offsets = NULL;
  double *upBorderLayer = NULL;
  double *downBorderLayer = NULL;
  double *prevLayer = NULL;
  double *currentLayer = NULL;
  MPI_Request send_up_req;
  MPI_Request send_down_req;
  MPI_Request recv_up_req;
  MPI_Request recv_down_req;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &processSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

  // Divide area
  layerHeights = new int[processSize];
  offsets = new int[processSize];
  setMatrixParts(layerHeights, offsets, processSize);

  // Init layers
  prevLayer = new double[layerHeights[processRank] * N_Y * N_Z];
  currentLayer = new double[layerHeights[processRank] * N_Y * N_Z];
  initLayers(prevLayer, currentLayer, layerHeights[processRank],
             offsets[processRank]);

  upBorderLayer = new double[N_Y * N_Z];
  downBorderLayer = new double[N_Y * N_Z];

  startTime = MPI_Wtime();

  do {
    double tmpMaxDiff = 0.0;
    double procMaxDelta = 0.0;

    std::swap(prevLayer, currentLayer);

    // Start sending and receiving border
    if (processRank != 0) {
      MPI_Isend(prevLayer, N_Y * N_Z, MPI_DOUBLE, processRank - 1, processRank,
                MPI_COMM_WORLD, &send_up_req);
      MPI_Irecv(upBorderLayer, N_Y * N_Z, MPI_DOUBLE, processRank - 1,
                processRank - 1, MPI_COMM_WORLD, &recv_up_req);
    }

    if (processRank != processSize - 1) {
      double *prev_down_border =
          prevLayer + (layerHeights[processRank] - 1) * N_Y * N_Z;
      MPI_Isend(prev_down_border, N_Y * N_Z, MPI_DOUBLE, processRank + 1,
                processRank, MPI_COMM_WORLD, &send_down_req);
      MPI_Irecv(downBorderLayer, N_Y * N_Z, MPI_DOUBLE, processRank + 1,
                processRank + 1, MPI_COMM_WORLD, &recv_down_req);
    }

    // Calculate center
    tmpMaxDiff = calcCenter(prevLayer, currentLayer, layerHeights[processRank],
                            offsets[processRank]);
    int send_up_completed = 0, recv_up_completed = 0;
    int send_down_completed = 0, recv_down_completed = 0;
    int flag1 = 0, flag2 = 0;

    if (processRank != 0 && processRank != processSize - 1) {
      while (!(send_up_completed && recv_up_completed && send_down_completed &&
               recv_down_completed)) {
        if (!send_up_completed || !recv_up_completed) {
          MPI_Test(&send_up_req, &send_up_completed, MPI_STATUS_IGNORE);
          MPI_Test(&recv_up_req, &recv_up_completed, MPI_STATUS_IGNORE);
        }
        if (send_up_completed && recv_up_completed && !flag1) {
          procMaxDelta = calcUpBorder(
              prevLayer, currentLayer, upBorderLayer, layerHeights[processRank],
              offsets[processRank], processRank, processSize);
          flag1 = 1;
        }
        if (!send_down_completed || !recv_down_completed) {
          MPI_Test(&send_down_req, &send_down_completed, MPI_STATUS_IGNORE);
          MPI_Test(&recv_down_req, &recv_down_completed, MPI_STATUS_IGNORE);
        }

        if (send_down_completed && recv_down_completed && !flag2) {
          procMaxDelta =
              calcDownBorder(prevLayer, currentLayer, downBorderLayer,
                             layerHeights[processRank], offsets[processRank],
                             processRank, processSize);
          flag2 = 1;
        }
      }
    }
    if (processRank == processSize - 1) {
      while (!(send_up_completed && recv_up_completed)) {
        MPI_Test(&send_up_req, &send_up_completed, MPI_STATUS_IGNORE);
        MPI_Test(&recv_up_req, &recv_up_completed, MPI_STATUS_IGNORE);
      }
      procMaxDelta = calcUpBorder(
          prevLayer, currentLayer, upBorderLayer, layerHeights[processRank],
          offsets[processRank], processRank, processSize);
    }
    if (processRank == 0) {
      while (!(send_down_completed && recv_down_completed)) {
        MPI_Test(&send_down_req, &send_down_completed, MPI_STATUS_IGNORE);
        MPI_Test(&recv_down_req, &recv_down_completed, MPI_STATUS_IGNORE);
      }
      procMaxDelta = calcDownBorder(
          prevLayer, currentLayer, downBorderLayer, layerHeights[processRank],
          offsets[processRank], processRank, processSize);
    }

    procMaxDelta = fmax(tmpMaxDiff, procMaxDelta);
    MPI_Allreduce(&procMaxDelta, &maxDiff, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
  } while (maxDiff >= EPSILON);

  // Calculate the differences of the calculated and theoretical functions
  maxDiff = calcMaxDiff(currentLayer, layerHeights[processRank],
                        offsets[processRank]);

  endTime = MPI_Wtime();

  if (processRank == 0) {
    printf("Time: %lf\n", endTime - startTime);
    printf("Max difference: %le\n", maxDiff);
  }

  delete[] offsets;
  delete[] layerHeights;
  delete[] prevLayer;
  delete[] currentLayer;
  delete[] upBorderLayer;
  delete[] downBorderLayer;

  MPI_Finalize();

  return EXIT_SUCCESS;
}

double rho(double x, double y, double z) { return 6 - A * phi(x, y, z); }

double phi(double x, double y, double z) { return x * x + y * y + z * z; }

int getIndex(int x, int y, int z) { return x * N_Y * N_Z + y * N_Z + z; }

double getX(int i) { return X_0 + i * H_X; }

double getY(int j) { return Y_0 + j * H_Y; }

double getZ(int k) { return Z_0 + k * H_Z; }

void setMatrixParts(int *layerHeights, int *offsets, int processSize) {
  int offset = 0, bufSize = N_X / processSize;
  int bufRemainder = N_X % processSize;
  for (int i = 0; i < processSize; ++i) {
    offsets[i] = offset;
    layerHeights[i] = bufSize;
    if (i < bufRemainder) ++layerHeights[i];
    offset += layerHeights[i];
  }
}

void initLayers(double *prevLayer, double *currentLayer, int layerHeight,
                int offset) {
  for (int i = 0; i < layerHeight; ++i)
    for (int j = 0; j < N_Y; j++)
      for (int k = 0; k < N_Z; k++) {
        bool isBorder = (offset + i == 0) || (j == 0) || (k == 0) ||
                        (offset + i == N_X - 1) || (j == N_Y - 1) ||
                        (k == N_Z - 1);
        if (isBorder) {
          prevLayer[getIndex(i, j, k)] =
              phi(getX(offset + i), getY(j), getZ(k));
          currentLayer[getIndex(i, j, k)] =
              phi(getX(offset + i), getY(j), getZ(k));
        } else {
          prevLayer[getIndex(i, j, k)] = 0;
          currentLayer[getIndex(i, j, k)] = 0;
        }
      }
}

double calcCenter(const double *prevLayer, double *currentLayer,
                  int layerHeight, int offset) {
  double phi_i;
  double maxDiff = 0.0;

  for (int i = 1; i < layerHeight - 1; ++i)
    for (int j = 1; j < N_Y - 1; ++j)
      for (int k = 1; k < N_Z - 1; ++k) {
        phi_i = (prevLayer[getIndex(i + 1, j, k)] +
                 prevLayer[getIndex(i - 1, j, k)]) /
                H_X_2;
        changeMaxDiff(phi_i, offset, maxDiff, i, j, k, prevLayer, currentLayer);
      }

  return maxDiff;
}

double calcUpBorder(const double *prevLayer, double *currentLayer,
                    const double *upBorderLayer, int layerHeight, int offset,
                    int processRank, int processSize) {
  double phi_i;
  double maxDiff = 0.0;

  for (int j = 1; j < N_Y - 1; ++j)
    for (int k = 1; k < N_Z - 1; ++k) {
      // Calculate the upper border
      if (processRank != 0) {
        int i = 0;
        phi_i = (prevLayer[getIndex(i + 1, j, k)] +
                 upBorderLayer[getIndex(0, j, k)]) /
                H_X_2;
        changeMaxDiff(phi_i, offset, maxDiff, i, j, k, prevLayer, currentLayer);
      }
    }
  return maxDiff;
}

double calcDownBorder(const double *prevLayer, double *currentLayer,
                      const double *downBorderLayer, int layerHeight,
                      int offset, int processRank, int processSize) {
  double phi_i;
  double maxDiff = 0.0;

  for (int j = 1; j < N_Y - 1; ++j)
    for (int k = 1; k < N_Z - 1; ++k) {
      // Calculate the lower border
      if (processRank != processSize - 1) {
        int i = layerHeight - 1;
        phi_i = (prevLayer[getIndex(i - 1, j, k)] +
                 downBorderLayer[getIndex(0, j, k)]) /
                H_X_2;
        changeMaxDiff(phi_i, offset, maxDiff, i, j, k, prevLayer, currentLayer);
      }
    }

  return maxDiff;
}

void changeMaxDiff(const double phi_i, int offset, double &maxDiff, int i,
                   int j, int k, const double *prevLayer,
                   double *currentLayer) {
  double phi_j, phi_k;
  double tmpMaxDiff = 0.0;
  phi_j =
      (prevLayer[getIndex(i, j + 1, k)] + prevLayer[getIndex(i, j - 1, k)]) /
      H_Y_2;
  phi_k =
      (prevLayer[getIndex(i, j, k + 1)] + prevLayer[getIndex(i, j, k - 1)]) /
      H_Z_2;
  currentLayer[getIndex(i, j, k)] =
      (phi_i + phi_j + phi_k - rho(getX(offset + i), getY(j), getZ(k))) /
      (2 / H_X_2 + 2 / H_Y_2 + 2 / H_Z_2 + A);

  // Check for calculation end
  tmpMaxDiff =
      fabs(currentLayer[getIndex(i, j, k)] - prevLayer[getIndex(i, j, k)]);
  if (tmpMaxDiff > maxDiff) maxDiff = tmpMaxDiff;
}

double calcMaxDiff(const double *currentLayer, int layerHeight, int offset) {
  double tmpMaxDelta = 0.0;
  double maxProcDelta = 0.0;
  double maxDelta = 0.0;

  for (int i = 0; i < layerHeight; ++i)
    for (int j = 0; j < N_Y; ++j)
      for (int k = 0; k < N_Z; ++k) {
        tmpMaxDelta = fabs(currentLayer[getIndex(i, j, k)] -
                           phi(getX(offset + i), getY(j), getZ(k)));
        if (tmpMaxDelta > maxProcDelta) maxProcDelta = tmpMaxDelta;
      }

  MPI_Allreduce(&maxProcDelta, &maxDelta, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  return maxDelta;
}