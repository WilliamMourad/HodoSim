#include "SteppingAction.hh"

#include "G4ProcessManager.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4OpticalPhoton.hh"
#include "G4MuonMinus.hh"
#include "G4SystemOfUnits.hh"

#include "OpticalPhotonTrackInfo.hh"
#include "MuTrackInfo.hh"


static inline G4bool isOpticalPhoton(const G4Track* track) {
	return track->GetDefinition() == G4OpticalPhoton::Definition();
}

static inline G4bool isPrimaryMuon(const G4Track* track) {
	return (track->GetDefinition() == G4MuonMinus::Definition()) && (track->GetParentID() == 0);
}


SteppingAction::SteppingAction(SteppingActionParameters steppingActionParameters) 
{
	_steppingActionParameters = steppingActionParameters;
};

SteppingAction::~SteppingAction() {};

void SteppingAction::UserSteppingAction(const G4Step* step) 
{
	const auto* track = step->GetTrack();

	ProcessOPReflections(track, step);
	ProcessMuPosition(track, step);
};


void SteppingAction::ProcessOPReflections(const G4Track* track, const G4Step* step)
{
	// Filter out non optical photons 
	// and for optical photons filter out steps that are not at a boundary
	if (!isOpticalPhoton(track)) return;
	
	auto preStep = step->GetPreStepPoint();
	
	if (preStep->GetStepStatus() != fGeomBoundary) return;

	static thread_local G4OpBoundaryProcess* boundary = nullptr;
	if (!boundary)
	{
		auto* processManager = track->GetDefinition()->GetProcessManager();

		auto* processVector = processManager->GetPostStepProcessVector(typeDoIt);

		for (size_t i = 0; i < processVector->size(); i++)
		{
			if (auto* b = dynamic_cast<G4OpBoundaryProcess*>((*processVector)[i]))
			{
				boundary = b;
				break;
			}
		}
		// This shouldnt happen but just in case...
		if (!boundary) return;
	}

	const auto status = boundary->GetStatus();

	// To determine the enumerator i just checked the source code of Geant4
	const bool isReflection =
		(status == G4OpBoundaryProcessStatus::TotalInternalReflection) ||
		(status == G4OpBoundaryProcessStatus::FresnelReflection) ||
		(status == G4OpBoundaryProcessStatus::LobeReflection) ||
		(status == G4OpBoundaryProcessStatus::SpikeReflection) ||
		(status == G4OpBoundaryProcessStatus::LambertianReflection) ||
		(status == G4OpBoundaryProcessStatus::BackScattering);

	if (!isReflection) return;

	auto trackInfo = static_cast<OpticalPhotonTrackInfo*>(track->GetUserInformation());

	if (!trackInfo) return;

	trackInfo->nReflections++;
	// Some logic can be added to check the volumes and determine if the reflection happened at a coating
	// but since the simulation setup is trivial (the result is the same), for now i'll just skip it
	trackInfo->nReflectionsAtCoating++;
}

void SteppingAction::ProcessMuPosition(const G4Track* track, const G4Step* step)
{
	// Filter out non primary muons
	// and for primary muons filter out steps that are not at a boundary
	if (!isPrimaryMuon(track)) return;

	auto preStep = step->GetPreStepPoint();
	auto postStep = step->GetPostStepPoint();
	
	if (preStep->GetStepStatus() != fGeomBoundary) return;
	
	auto* postPV = postStep->GetPhysicalVolume();
	if (!postPV) return;

	if (postPV->GetLogicalVolume()->GetName() != _steppingActionParameters.scintLVName) return;


	auto* trackInfo = static_cast<MuTrackInfo*>(track->GetUserInformation());

	if (!trackInfo) return;
	
	// It is also possible to store the momentum direction here but i dont need it for now
	// const auto momentumDir = postStep->GetMomentumDirection();
	const auto globalPos = postStep->GetPosition();
	const auto localPos = postStep->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(globalPos);
	const auto globalTime = track->GetGlobalTime();

	// I added this debug line just to make sure everything is working as intended
	// G4cout << "[Stepping Action] MuonXPos: " << globalPos.x() / cm << " cm" << G4endl;

	trackInfo->enteredScint = true;
	trackInfo->globalEntryPosition = globalPos;
	trackInfo->localEntryPosition = localPos;
	trackInfo->globalTime = globalTime;
}