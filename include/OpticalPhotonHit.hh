#pragma once

#include "G4VHit.hh"
#include "G4Allocator.hh"

#include "G4ThreeVector.hh"

class OpticalPhotonHit : public G4VHit
{
public:
	OpticalPhotonHit();
	~OpticalPhotonHit();

	inline void* operator new(size_t);
	inline void operator delete(void* hit);
	
	void SetEventID(G4int eventID);
	void SetEdep(G4double edep);
	void SetProcess(G4String process);
	void SetPosition(G4ThreeVector position);
	void SetTime(G4double time);
	void SetNReflections(G4int nReflections);
	void SetNReflectionsAtCoating(G4int nReflectionsAtCoating);


	G4double GetEdep() const { return _edep; }
	G4String GetProcess() const { return _process; }
	G4double GetTime() const { return _time; }
	G4ThreeVector GetPosition() const { return _position; }
	G4int GetNReflections() const { return _nReflections; }
	G4int GetNReflectionsAtCoating() const { return _nReflectionsAtCoating; }

private:
	G4int _eventID;
	G4double _edep;
	G4String _process;
	G4double _time;
	G4ThreeVector _position;
	G4int _nReflections;
	G4int _nReflectionsAtCoating;
};

// Memory allocation handler (required for hits collections)
extern G4ThreadLocal G4Allocator<OpticalPhotonHit>* OpticalPhotonHitAllocator;

inline void* OpticalPhotonHit::operator new(size_t)
{
	if (!OpticalPhotonHitAllocator) OpticalPhotonHitAllocator = new G4Allocator<OpticalPhotonHit>;
	return (void*)OpticalPhotonHitAllocator->MallocSingle();
}

inline void OpticalPhotonHit::operator delete(void* hit)
{
	OpticalPhotonHitAllocator->FreeSingle((OpticalPhotonHit*)hit);
}