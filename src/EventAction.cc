#include "EventAction.hh"

#include "G4HCofThisEvent.hh"
#include "G4SystemOfUnits.hh"

#include "OpticalPhotonHit.hh"


EventAction::EventAction(EventActionParameters eventActionParameters) 
{
	_eventActionParameters = eventActionParameters;
	analysisManager = G4AnalysisManager::Instance();
}

EventAction::~EventAction() {}

void EventAction::BeginOfEventAction(const G4Event* event)
{
	
}

void EventAction::EndOfEventAction(const G4Event* event) 
{
	muonHitRegistered = false; // reset for next event

	G4String siliconPMSDName = _eventActionParameters.siliconPMSDName;

	G4String opCName = _eventActionParameters.opCName;

	G4HCofThisEvent* hce = event->GetHCofThisEvent();

	auto SDManager = G4SDManager::GetSDMpointer();

	if (!hce) return;

	// Get HCID once 
	if (siliconPM_op_HCID < 0)
	{
		siliconPM_op_HCID = SDManager->GetCollectionID(opCName); // good

		// Querying the HCID this way is the best way to ask for errors,
		// I'll just let it slide this time (maybe i'll fix it later)
		scint_edep_HCID = SDManager->GetCollectionID("ScintillatorMFD/Edep"); // bad
		scint_muPathLength_HCID = SDManager->GetCollectionID("ScintillatorMFD/MuPathLength"); // bad
		coating_edep_HCID = SDManager->GetCollectionID("CoatingMFD/Edep"); // bad
	}

	auto siliconPMSD_HC = hce->GetHC(siliconPM_op_HCID);
	auto scint_edep_HC = hce->GetHC(scint_edep_HCID);
	auto scint_muPathLength_HC = hce->GetHC(scint_muPathLength_HCID);
	auto coating_edep_HC = hce->GetHC(coating_edep_HCID);
	
	auto* map_scint_edep_HC = static_cast<G4THitsMap<G4double>*>(scint_edep_HC);
	auto* map_scint_muPathLength_HC = static_cast<G4THitsMap<G4double>*>(scint_muPathLength_HC);
	auto* map_coating_edep_HC = static_cast<G4THitsMap<G4double>*>(coating_edep_HC);
	
	// From here on, i'll just fill the root structures with the data

	const G4int nSiPMs = 4; // hardcoded for now, will be improved later
	G4int nScintHits[nSiPMs] = {0, 0, 0, 0};
	G4int nCerHits[nSiPMs] = {0, 0, 0, 0};
	G4double scintEdep = SumOverHC(map_scint_edep_HC);
	G4double scintMuPathLength = SumOverHC(map_scint_muPathLength_HC);
	G4double coatingEdep = SumOverHC(map_coating_edep_HC);
	G4double muonHitX = muonLocalEntryPosition.x();
	G4double muonHitY = muonLocalEntryPosition.y();

	// Analyze & Store in Histograms
	#pragma region Histograms

	if (siliconPMSD_HC) {
		G4int nHits = siliconPMSD_HC->GetSize();
		

		for (G4int i = 0; i < nHits; i++) {
			auto hit = static_cast<OpticalPhotonHit*>(siliconPMSD_HC->GetHit(i));
			auto process = hit->GetProcess();
			auto edep = hit->GetEdep();
			auto time = hit->GetTime();
			auto position = hit->GetPosition();
			auto nReflections = hit->GetNReflections();
			auto nReflectionsAtCoating = hit->GetNReflectionsAtCoating();
			auto siPMID = hit->GetSiPMID();

			// Check siPMID validity
			if (siPMID < 0 || siPMID >= nSiPMs) continue;

			// dont forget to remove the g4 units
			if (process == "Scintillation") {
				analysisManager->FillH1(0, edep / eV); // Scint OP Energy 
				analysisManager->FillH1(1, time / ns); // Scint OP Time
				analysisManager->FillH2(0, position.x() / mm, position.y() / mm); // Scint OP Spread
				analysisManager->FillH1(2 + siPMID, nReflectionsAtCoating); // OP Reflections
				nScintHits[siPMID]++;
			} else if (process == "Cerenkov") {
				nCerHits[siPMID]++;
			}

		}
	}

	#pragma endregion Histograms

	// Analyze & Store in NTuples
	#pragma region Ntuples
		
	if (siliconPMSD_HC && scint_edep_HC && scint_muPathLength_HC && coating_edep_HC)
	{
		analysisManager->FillNtupleDColumn(0, event->GetEventID());		// eventID
		analysisManager->FillNtupleDColumn(1, nScintHits[0]);			// scint OP hits
		analysisManager->FillNtupleDColumn(2, nScintHits[1]);			// scint OP hits
		analysisManager->FillNtupleDColumn(3, nScintHits[2]);			// scint OP hits
		analysisManager->FillNtupleDColumn(4, nScintHits[3]);			// scint OP hits
		analysisManager->FillNtupleDColumn(5, nCerHits[0]);				// cer OP hits
		analysisManager->FillNtupleDColumn(6, nCerHits[1]);				// cer OP hits
		analysisManager->FillNtupleDColumn(7, nCerHits[2]);				// cer OP hits
		analysisManager->FillNtupleDColumn(8, nCerHits[3]);				// cer OP hits
		analysisManager->FillNtupleDColumn(9, scintEdep / eV);			// scint edep
		analysisManager->FillNtupleDColumn(10, coatingEdep / eV);		// coating edep
		analysisManager->FillNtupleDColumn(11, scintMuPathLength / mm);	// scint mu path length
		analysisManager->FillNtupleDColumn(12, muonHitX / mm);			// muon X coordinate on hit
		analysisManager->FillNtupleDColumn(13, muonHitY / mm);			// muon Y coordinate on hit 
		analysisManager->AddNtupleRow();
	}

	#pragma endregion Ntuples
}

void EventAction::RegisterMuonHit(G4ThreeVector localPos, G4ThreeVector globalPos, G4double tGlob)
{
	// For starting I will assume that only one muon is present per event.
	// Therefore this logic will need to be revised in case of multiple muons.
	// To avoid errors in such a scenario, I will always sample just the first muon hit.
	if (muonHitRegistered) return;
	
	muonHitRegistered = true;
	muonLocalEntryPosition = localPos;
	muonGlobalEntryPosition = globalPos;
	muonGlobalTime = tGlob;
}

G4double EventAction::SumOverHC(const G4THitsMap<G4double>* hm)
{
	G4double sum = 0.;
	if (!hm) return 0.;
	for (const auto& kv : *hm->GetMap()) sum += *(kv.second);
	return sum;
}