#pragma once

#include "G4UserTrackingAction.hh"


// Forward declaration
class EventAction; 


struct TrackingActionParameters {};

class TrackingAction : public G4UserTrackingAction
{
public:
	TrackingAction(TrackingActionParameters trackingActionParameters, EventAction* eventAction);
	~TrackingAction();

	void PreUserTrackingAction(const G4Track* track) override;
	void PostUserTrackingAction(const G4Track* track) override;

private:
	TrackingActionParameters _trackingActionParameters;
	EventAction* _eventAction = nullptr;
};