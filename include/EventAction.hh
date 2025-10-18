#pragma once

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "G4THitsMap.hh"
#include "G4SDManager.hh"
#include "G4AnalysisManager.hh"


struct EventActionParameters {
	G4String scintLVName;
	G4String siliconPMSDName;
	G4String opCName;
};

class EventAction : public G4UserEventAction {
public:
	EventAction(EventActionParameters eventActionParameters);
	~EventAction();

	void BeginOfEventAction(const G4Event* event) override;
	void EndOfEventAction(const G4Event* event) override;

	// I defined this method that will be called from the TrackingAction
	// to register the position of the muon when it enters the scintillator.
	// It can be expanded later to include other information if needed.
	void RegisterMuonHit(G4ThreeVector localPos, G4ThreeVector globalPos, G4double tGlob);

private:

	static G4double SumOverHC(const G4THitsMap<G4double>* hm);
	
	EventActionParameters _eventActionParameters;
	G4AnalysisManager* analysisManager;

	// Muon data for the current event
	// For starting I will assume that only one muon is present per event.
	// Therefore this logic will need to be revised in case of multiple muons.
	G4bool muonHitRegistered = false;
	G4ThreeVector muonLocalEntryPosition = G4ThreeVector(0., 0., 0.);
	G4ThreeVector muonGlobalEntryPosition = G4ThreeVector(0., 0., 0.);
	G4double muonGlobalTime = 0.;

	// Cache hit collections IDs to improve performances
	G4int siliconPM_op_HCID			= -1;
	G4int scint_edep_HCID			= -1;
	G4int scint_muPathLength_HCID	= -1;
	G4int coating_edep_HCID			= -1;
	G4int siliconPM_edep_HCID		= -1;
};
