#include "OpticalPhotonHit.hh"

OpticalPhotonHit::OpticalPhotonHit() {}

OpticalPhotonHit::~OpticalPhotonHit() {}


void OpticalPhotonHit::SetEventID(G4int eventID)
{
	_eventID = eventID;
}

void OpticalPhotonHit::SetEdep(G4double edep)
{
	_edep = edep;
}

void OpticalPhotonHit::SetProcess(G4String process)
{
	_process = process;
}

void OpticalPhotonHit::SetTime(G4double time)
{
	_time = time;
}

void OpticalPhotonHit::SetPosition(G4ThreeVector position)
{
	_position = position;
}

void OpticalPhotonHit::SetNReflections(G4int nReflections)
{
	_nReflections = nReflections;
}

void OpticalPhotonHit::SetNReflectionsAtCoating(G4int nReflectionsAtCoating)
{
	_nReflectionsAtCoating = nReflectionsAtCoating;
}


G4ThreadLocal G4Allocator<OpticalPhotonHit>* OpticalPhotonHitAllocator = nullptr;

