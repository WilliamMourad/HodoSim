#include "YAMLParser.hh"


YAMLParser::YAMLParser(const char* filename) {
	
	_filename = filename;

	content = slurp();
	tree = ryml::parse_in_place(ryml::to_substr(content));
	root = tree.rootref();
}

YAMLParser::~YAMLParser() {}


std::string YAMLParser::slurp() {
	// To avoid possible issues with newlines on different platforms
	// the safest way is to read the file in binary mode
	std::ifstream f(_filename, std::ios::binary);
	if (!f)
	{
		loaded = false;
		throw std::runtime_error(std::string("[YAMLParser] Could not open file: ") + _filename);
	};
	std::ostringstream ss;
	ss << f.rdbuf();
	loaded = true;
	return ss.str();
}

// This is to get a child in a safe way, throwing an error if the key is missing	
ryml::NodeRef YAMLParser::require(ryml::NodeRef parent, const char* key) {
	
	if (!parent.has_child(key)) {
		throw std::runtime_error(std::string("[YAMLParser] Missing required key: ") + key);
	}
	return parent[key];
}

// These are just convenience functions to convert node values to basic types
// I skipped error handling for brevity (I'll 100% add it one day)
// be careful not to misuse them.
double YAMLParser::as_double(ryml::NodeRef node) {
	auto s = node.val();
	return std::stod(std::string(s.str, s.len));
}

int YAMLParser::as_int(ryml::NodeRef node) {
	auto s = node.val();
	return std::stoi(std::string(s.str, s.len));
}

std::string YAMLParser::as_string(ryml::NodeRef node) {
	auto s = node.val();
	return std::string(s.str, s.len);
}

bool YAMLParser::as_bool(ryml::NodeRef node) {
	auto s = node.val();
	std::string valStr(s.str, s.len);
	if (valStr == "true" || valStr == "1") {
		return true;
	}
	else if (valStr == "false" || valStr == "0") {
		return false;
	}
	else {
		throw std::runtime_error(std::string("[YAMLParser] Invalid boolean value: ") + valStr);
	}
}