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

	const G4int nSiPMs = _eventActionParameters.sipmsPerSide * 4;
	std::vector<G4int> nScintHits(nSiPMs);
	std::vector<G4int> nCerHits(nSiPMs);
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
			auto siPMID = hit->GetSiPMID();

			// Check siPMID validity
			if (siPMID < 0 || siPMID >= nSiPMs) continue;

			// dont forget to remove the g4 units
			if (process == "Scintillation") {
				analysisManager->FillH1(0, edep / eV); // Scint OP Energy 
				analysisManager->FillH1(1, time / ns); // Scint OP Time
				analysisManager->FillH2(0, position.x() / mm, position.y() / mm); // Scint OP Spread
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
		// eventID
		analysisManager->FillNtupleDColumn(0, event->GetEventID());		
		
		// scint OP hits
		for (int i = 0; i < nSiPMs; i++)
		{
			analysisManager->FillNtupleDColumn(1 + i, nScintHits[i]);

		}
		// cer OP hits
		for (int i = 0; i < nSiPMs; i++)
		{
			analysisManager->FillNtupleDColumn(1 + nSiPMs + i, nCerHits[i]);
		}

		G4int ct = 1 + 2 * nSiPMs;
		analysisManager->FillNtupleDColumn(ct, scintEdep / eV);			// scint edep
		analysisManager->FillNtupleDColumn(ct + 1, coatingEdep / eV);		// coating edep
		analysisManager->FillNtupleDColumn(ct + 2, scintMuPathLength / mm);	// scint mu path length
		analysisManager->FillNtupleDColumn(ct + 3, muonHitX / mm);			// muon X coordinate on hit
		analysisManager->FillNtupleDColumn(ct + 4, muonHitY / mm);			// muon Y coordinate on hit 
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