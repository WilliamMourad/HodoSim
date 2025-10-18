#pragma once

#include "G4VSensitiveDetector.hh"
#include "G4THitsCollection.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Step.hh"

#include "OpticalPhotonHit.hh"


class SiliconPMSD : public G4VSensitiveDetector
{
public:
	SiliconPMSD(const G4String& name, G4String cName);
	~SiliconPMSD();

	void Initialize(G4HCofThisEvent* hce) override;
	G4bool ProcessHits(G4Step* step, G4TouchableHistory* history) override;
	void EndOfEvent(G4HCofThisEvent* hce) override;

private:
	G4String _cName;
	G4THitsCollection<OpticalPhotonHit>* opHitsCollection;
	G4int hcID; // cache the hit collection ID to improve performances
};