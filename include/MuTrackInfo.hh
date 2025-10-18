#pragma once

#include "G4VUserTrackInformation.hh"
#include "G4ThreeVector.hh"

// I'll use this class to track the position of muons once they reach the scintillator.
// This way I'll be able to evaluate the efficiency of the beam reconstruction algorithm 
// tha I'll implement in the full hodoscope simulation.
class MuTrackInfo : public G4VUserTrackInformation
{
public:
	MuTrackInfo() = default;

	G4bool enteredScint = false;
	G4ThreeVector globalEntryPosition;	// global and local refers to the scintillator frame of reference
	G4ThreeVector localEntryPosition;
	G4double globalTime = 0.;
};

