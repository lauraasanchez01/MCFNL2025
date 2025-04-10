import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

# Constants for permittivity regions test
EPS1 = 2.0
C1 = 1 / np.sqrt(MU0*EPS1)
R = (np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T = 2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def sigmoid_grid(xmin=-1, xmax=1, npoints=101, steepness=5, midpoint=0): # midpoint=0 for centered sigmoid in interval
  """
  Generates a non-uniform grid using a sigmoid function.

  """
  x = np.linspace(-steepness, steepness, npoints)
  grid = xmin + (xmax - xmin) / (1 + np.exp(-x + midpoint*steepness)) #Sigmoid function
  grid = xmin + (grid - np.min(grid)) / (np.max(grid) - np.min(grid)) * (xmax - xmin) #Rescale it so that it matches the endpoints
  return grid



class FDTD1D:
    def __init__(self, xE, bounds=('pec', 'pec')):
        self.xE = np.array(xE)
        self.xH = (self.xE[:-1] + self.xE[1:]) / 2.0
        self.dxE = np.diff(self.xE)
        self.avgdxE = np.mean(self.dxE) # For energy calculations
        self.dxH = np.diff(self.xH)
        self.dt = 0.9 * np.min(np.concatenate([self.dxE, self.dxH])) / C0 # Choose a safe dt by default for stability
        self.bounds = bounds
        self.e = np.zeros_like(self.xE)
        self.h = np.zeros_like(self.xH)
        self.h_old = np.zeros_like(self.h)
        self.eps = np.ones_like(self.xE)  # Default permittivity is 1 everywhere
        self.initialized = False
        self.tfsf = False
        self.energyE = []
        self.energyH = []
        self.energy = []

    def set_initial_condition(self, initial_condition, initial_h_condition=None):
        self.e[:] = initial_condition[:]
        if initial_h_condition is not None:
            self.h[:] = initial_h_condition[:]
        self.initialized = True

    def set_permittivity_regions(self, regions):
        """Set different permittivity regions in the grid.

        Args:
            regions: List of tuples (start_x, end_x, eps_value) defining regions
                    with different permittivity values.
        """
        for start_x, end_x, eps_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.eps[start_idx:end_idx] = eps_value

        max_eps = np.max(self.eps)
        c_max = 1 / np.sqrt(MU0*max_eps)
        self.dt = 0.9 * np.min(np.concatenate([self.dxE, self.dxH])) / c_max  # Redefine the safe dt according to permittivities

    def set_tfsf_conditions(self, x_start, x_end, function):
        self.x_start = x_start
        self.x_end = x_end
        self.tfsolver = FDTD1D(self.xE, bounds=('mur', 'mur'))
        self.tfsolver.set_initial_condition(function)
        self.tfsf = True

    def step(self):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]

        self.h[:] = self.h[:] - self.dt / self.dxE[:] / MU0 * (self.e[1:] - self.e[:-1])
        self.e[1:-1] = self.e[1:-1] - self.dt / self.dxH[:] / self.eps[1:-1] * (self.h[1:] - self.h[:-1])

        # Bound calculation

        if self.bounds[0] == 'pec':
            self.e[0] = 0.0
        elif self.bounds[0] == 'mur':
            self.e[0] = self.e_old_left + (C0*self.dt - self.dxE[0]) / \
                (C0*self.dt + self.dxE[0])*(self.e[1] - self.e[0])
        elif self.bounds[0] == 'pmc':
            self.e[0] = self.e[0] - 2 * self.dt/ self.dxE[0]/ EPS0*(self.h[0])
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[0]}")

        if self.bounds[1] == 'pec':
            self.e[-1] = 0.0
        elif self.bounds[1] == 'mur':
            self.e[-1] = self.e_old_right + (C0*self.dt - self.dxE[-1]) / \
                (C0*self.dt + self.dxE[-1])*(self.e[-2] - self.e[-1])
        elif self.bounds[1] == 'pmc':
            self.e[-1] = self.e[-1] + 2 * self.dt/self.dxE[-1] / EPS0*(self.h[-1])
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[1]}")

        # Energy calculation
        self.energyE.append(0.5 * np.dot(self.e, self.avgdxE * self.eps * self.e))
        self.energyH.append(0.5 * np.dot(self.h_old, self.avgdxE * MU0 * self.h))
        self.energy.append(0.5 * np.dot(self.e, self.avgdxE * self.eps * self.e) + 0.5 * np.dot(self.h_old, self.avgdxE * MU0 * self.h))
        self.h_old[:] = self.h[:]

    def run_until(self, Tf=None, dt=None, n_steps=100):
        if not self.initialized:
            raise RuntimeError(
                "Initial condition not set. Call set_initial_condition first.")

        if dt is not None: # Define a specific dt for the solver
            if dt > self.dt:
              raise RuntimeError(
                  "Too high dt value. Method is not stable. Set it to a lower value")
            else:
              self.dt = dt

        if Tf is not None: # If a final time is defined, let it run until then. If not, use the steps.
          used_n_steps = int(Tf / self.dt)
          for n in range(used_n_steps):
            self.step()

            #TFSF calculation
            if self.tfsf:
              self.tfsolver.step()
              self.e[(np.abs(self.xE - self.x_start)).argmin()] += self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_start)).argmin()]
              self.e[(np.abs(self.xE - self.x_end)).argmin()] -= self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_end)).argmin()]
              self.h[(np.abs(self.xH - self.x_start)).argmin()] += self.tfsolver.h[(np.abs(self.tfsolver.xH - self.x_start)).argmin()]
              self.h[(np.abs(self.xH - self.x_end)).argmin()] -= self.tfsolver.h[(np.abs(self.tfsolver.xH - self.x_end)).argmin()]

        else:
          for n in range(n_steps):
            self.step()

            # TFSF calculation
            if self.tfsf:
              self.tfsolver.step()
              self.e[(np.abs(self.xE - self.x_start)).argmin()] += self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_start)).argmin()]
              self.e[(np.abs(self.xE - self.x_end)).argmin()] -= self.tfsolver.e[(np.abs(self.tfsolver.xE - self.x_end)).argmin()]

        return self.e
