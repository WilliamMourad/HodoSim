#include <ryml.hpp>
#include <c4/yml/std/string.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>


// For starting I'll keep this parser class very simple.
// I will expand it adding a proper error handling later.
class YAMLParser {
public:
	YAMLParser(const char* filename);
	~YAMLParser();

	static ryml::NodeRef require(ryml::NodeRef parent, const char* key);
	
	static double as_double(ryml::NodeRef node);
	static int as_int(ryml::NodeRef node);
	static std::string as_string(ryml::NodeRef node);
	static bool as_bool(ryml::NodeRef node);

	ryml::NodeRef getRoot() const { return root; };
	bool isLoaded() const { return loaded; }
	
private:
	std::string slurp();

	const char* _filename;
	
	std::string content;
	ryml::Tree tree;
	ryml::NodeRef root;

	bool loaded = false;
};