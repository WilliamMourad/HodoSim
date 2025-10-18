#pragma once

#include "G4VSDFilter.hh"
#include "G4Track.hh"
#include "G4MuonMinus.hh"

// I'll use this to filter muons interacting with the detector
// to sample edep / pathLength data with MFDs.
// To learn more on how it works check out https://indico.cern.ch/event/58317/contributions/2047432/attachments/992284/1411033/Scoring2.pdf

class PrimaryMuonFilter : public G4VSDFilter
{
public:
	// (for now) I'll keep the definitions in the header
	PrimaryMuonFilter(const G4String& name) : G4VSDFilter(name) {}
	G4bool Accept(const G4Step* step) const override {
		const auto* track = step->GetTrack();
		const auto* particleDefinition = track->GetDefinition();

		// Filter out non primaries
		if (track->GetParentID() != 0) return false;

		// Eventually I could also allow muPlus particles
		return (particleDefinition == G4MuonMinus::MuonMinusDefinition());
	}
};