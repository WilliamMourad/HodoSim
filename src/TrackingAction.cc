#include "TrackingAction.hh"
#include "EventAction.hh"
#include "OpticalPhotonTrackInfo.hh"
#include "MuTrackInfo.hh"

#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4MuonMinus.hh"
#include "G4SystemOfUnits.hh"


static inline G4bool isOpticalPhoton(const G4Track* track) {
	return track->GetDefinition() == G4OpticalPhoton::Definition();
}

static inline G4bool isPrimaryMuon(const G4Track* track) {
	return (track->GetDefinition() == G4MuonMinus::Definition()) && (track->GetParentID() == 0);
}


TrackingAction::TrackingAction(TrackingActionParameters trackingActionParameters, EventAction* eventAction) 
{
	_trackingActionParameters = trackingActionParameters;
	_eventAction = eventAction;
};

TrackingAction::~TrackingAction() {};

void TrackingAction::PreUserTrackingAction(const G4Track* track) 
{
	// I want to track optical photons to sample the number of reflections they undergo
	if (isOpticalPhoton(track))
	{
		if (!track->GetUserInformation())
		{
			const_cast<G4Track*>(track)->SetUserInformation(new OpticalPhotonTrackInfo());
		}
	}
	// I want to track primary muons to register their position in the scintillator
	if (isPrimaryMuon(track))
	{
		if (!track->GetUserInformation())
		{
			const_cast<G4Track*>(track)->SetUserInformation(new MuTrackInfo());
		}
	}
};

void TrackingAction::PostUserTrackingAction(const G4Track* track) 
{
	if (!isPrimaryMuon(track)) return;
	auto* trackInfo = static_cast<MuTrackInfo*>(track->GetUserInformation());
	
	if (!trackInfo || !_eventAction) return;

	if (trackInfo->enteredScint)
	{
		// I added this debug line just to make sure everything is working as intended
		// G4cout << "[Tracking Action] MuonXPos: " << trackInfo->globalEntryPosition.x() / cm << " cm" << G4endl;

		_eventAction->RegisterMuonHit(
			trackInfo->localEntryPosition,
			trackInfo->globalEntryPosition,
			trackInfo->globalTime
		);
	}
};