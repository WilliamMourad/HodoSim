#pragma once

#include "G4VUserTrackInformation.hh"

// I'll use this class to keep track of the number of reflections each optical photon undergoes in its lifetime.
// This is not strictly necessary but i think it's a powerful metric for an optimization study of the detector geometry.
class OpticalPhotonTrackInfo : public G4VUserTrackInformation
{
public:
	OpticalPhotonTrackInfo() = default;

	// In theory the OP reflects only at coatings
	// I made this distinction to account for an eventual future scenario in which optical grease or an optical guide is used.
	int nReflections = 0;
	int nReflectionsAtCoating = 0;
};

