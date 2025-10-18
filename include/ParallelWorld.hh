#pragma once

#include "G4VUserParallelWorld.hh"
#include "G4LogicalVolume.hh"


// I'll use a parallel world to capture the edep in the SiPM,
// this way i can keep the keep optics and edep detection logic separated
// (I know it's more code but i really believe that modularity wins in the long run)
class ParallelWorld : public G4VUserParallelWorld {

public:
	ParallelWorld(const G4String& name, G4double plateThickness, G4double siPMThickness, G4double gap);
	~ParallelWorld();

	void Construct();
	void ConstructSD();

private:
	G4double _plateThickness;
	G4double _siPMThickenss;
	G4double _gap;
	G4LogicalVolume* fGhostSiPMLV = nullptr;
};