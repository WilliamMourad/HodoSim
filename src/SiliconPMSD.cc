#include "SiliconPMSD.hh"

#include "G4SDManager.hh"
#include "G4AnalysisManager.hh"
#include "G4OpticalPhoton.hh" // not to be confused with OpticalPhotonHit.hh
#include "G4RunManager.hh"

#include "OpticalPhotonTrackInfo.hh"


SiliconPMSD::SiliconPMSD(const G4String& name, G4String cName) : G4VSensitiveDetector(name), opHitsCollection(nullptr)
{
	// collectionName is a variable of G4VSensitiveDetector

	_cName = cName;
	collectionName.insert(_cName);
}

SiliconPMSD::~SiliconPMSD() {}

void SiliconPMSD::Initialize(G4HCofThisEvent* hce)
{
	// SensitiveDetectorName is a variable of G4VSensitiveDetector
	opHitsCollection = new G4THitsCollection<OpticalPhotonHit>(SensitiveDetectorName, collectionName[0]);

	hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
	hce->AddHitsCollection(hcID, opHitsCollection);
}

G4bool SiliconPMSD::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
	G4Track* track = step->GetTrack();
	G4int eventID = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

	// Filter out non optical photons
	if (track->GetDefinition() != G4OpticalPhoton::OpticalPhotonDefinition()) return false;

	// Get the process that created the optical photon
	G4String creatorProcess = "none";
	if (track->GetCreatorProcess() != nullptr) {
		creatorProcess = track->GetCreatorProcess()->GetProcessName();
	
		// Just to check if something weird is happening
		if (creatorProcess != "Scintillation" && creatorProcess != "Cerenkov") {
			G4cout << "----------------------------------" << G4endl;
			G4cout << "----------------------------------" << G4endl;
			G4cout << "WARNING - unrecognized process: " << creatorProcess << G4endl;
			G4cout << "----------------------------------" << G4endl;
			G4cout << "----------------------------------" << G4endl;
	
			return false;
	
		}
	}

	// Retrieve info (reflections) from the track user information
	auto* trackInfo = static_cast<OpticalPhotonTrackInfo*>(track->GetUserInformation());
	G4int nReflections = trackInfo ? trackInfo->nReflections : 0;
	G4int nReflectionsAtCoating = trackInfo ? trackInfo->nReflectionsAtCoating : 0;


	OpticalPhotonHit* opHit = new OpticalPhotonHit();
	opHit->SetEventID(eventID);
	opHit->SetEdep(step->GetTotalEnergyDeposit());
	opHit->SetProcess(creatorProcess);
	opHit->SetTime(track->GetGlobalTime());
	opHit->SetPosition(step->GetPreStepPoint()->GetPosition());
	opHit->SetNReflections(nReflections);
	opHit->SetNReflectionsAtCoating(nReflectionsAtCoating);

	// add the number of reflections of the optical photon with the coating surface 

	opHitsCollection->insert(opHit);


	// kill the track setting G4TrackStatus=fStopAndKill
	track->SetTrackStatus(fStopAndKill);

	return true;
}

void SiliconPMSD::EndOfEvent(G4HCofThisEvent* hce)
{
	// G4int hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
	auto hc = static_cast<G4THitsCollection<OpticalPhotonHit>*>(hce->GetHC(hcID));

	// The following block is for debug purposes
	// if (!hc)
	// {
	// 	G4cerr << "[SD] Could not find hits collection with ID" << hcID << G4endl;
	// 	return;
	// }
	// 
	// G4cout << "[SD] Number of Optical Photons Hits Collected by SiPMSD: " << hc->entries() << G4endl;
	
	return;
}