# Purposes
---
A Geant4 simulation of an hodoscope, used to monitor the properties of a muon beam.
The Scint+SiPM detector used in the simulation comes from the MuScinTest project, where it was accurately studied and verified.

In particular, for a fixed set of parameters such as 
- the configuration of the components of the hodoscope,
- the geometry of single components,
- the properties of the incoming beam


(primarily) i want to achieve 
- a good prediction power over the beam position/shape (and eventually its intensity) using the data collected by the hodoscope.
- a certain transparency to muons of different energies (to minimize the impact of the hodoscope on the beam itself).
- a consistent equivalence with real experimental data (once available).


# Implementation
---
With that in mind, there are multiple configurations that can be studied. 

## Single Plate Configuration

the first one i want to implement is the following: a single scintillator plate (BC-412) of size 50mm x 50mm x 3mm, coated on the front and back sides with a reflective layer 
(Aluminum) of thickness 50um, surrounded by 4 SiPM (50mm x 3mm x 3mm), one for each lateral side of the plate.

This configuration's pro are that:
- relying on a single scintillator plate should lower the cost and complexity of the installation,
- the muons cross a minimal amount of material (thus losing less energy),
- the 4 SiPM should provide a good prediction power over the beam position.

The Cons are that:
- the scintillator plate thickness is still bounded by the SiPMs thickness (1-3mm),
- the information on the beam intensity cant be isolated from the background.



# Observations  
---

## On Physics

## On Geant4

# Questions
---


# Output
---
Before prooceding i want to report some important considerations regarding data.

- A lot of space is required to sample single optical photons information (per-photon logs) (just imagine that for each MeV of the crossing particle, approx 10k optical photons are produced, 
the sim requires tens of thousands of muons to store enough data for other plots).
To avoid huge output files i suggest using histograms (and not NTuples) to sample optical photons.

All the collected data is already listed in the MuScinTest project description, check that for reference.
Eventually (if needed) i'll gather some more data and report it here.

# Next Step
---
Each configuration proposed can undergo an optimization process using machine learning techniques (bayesian optimization, reinforcement learning, etc..).
This will not be covered in this project but in the future i may explore this path.

# History of Changes
---
- 2025-10-18: Initial commit, initialized the project starting from a clone of the MuScinTest project.
