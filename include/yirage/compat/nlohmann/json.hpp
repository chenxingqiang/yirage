#ifndef NLOHMANN_JSON_COMPAT_HPP
#define NLOHMANN_JSON_COMPAT_HPP

/*
 * nlohmann/json compatibility header for builds without the library
 * Provides basic JSON functionality stub when nlohmann/json is not available
 */

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <initializer_list>
#include <type_traits>

namespace nlohmann {

// Basic JSON value types
enum class json_value_type {
    null,
    object,
    array,
    string,
    boolean,
    number_integer,
    number_unsigned,
    number_float
};

// Simplified JSON class for compatibility
class json {
public:
    using object_t = std::map<std::string, json>;
    using array_t = std::vector<json>;
    using string_t = std::string;
    using boolean_t = bool;
    using number_integer_t = std::int64_t;
    using number_unsigned_t = std::uint64_t;
    using number_float_t = double;

private:
    json_value_type type_ = json_value_type::null;
    union {
        object_t* object;
        array_t* array;
        string_t* string;
        boolean_t boolean;
        number_integer_t number_integer;
        number_unsigned_t number_unsigned;
        number_float_t number_float;
    } value_{};

public:
    // Constructors
    json() = default;
    
    json(const std::string& str) {
        type_ = json_value_type::string;
        value_.string = new string_t(str);
    }
    
    json(const char* str) {
        type_ = json_value_type::string;
        value_.string = new string_t(str);
    }
    
    json(bool b) {
        type_ = json_value_type::boolean;
        value_.boolean = b;
    }
    
    json(int i) {
        type_ = json_value_type::number_integer;
        value_.number_integer = i;
    }
    
    json(double d) {
        type_ = json_value_type::number_float;
        value_.number_float = d;
    }
    
    // Constructor for vectors and other containers
    template<typename T>
    json(const std::vector<T>& vec) {
        type_ = json_value_type::array;
        value_.array = new array_t();
        for (const auto& item : vec) {
            value_.array->push_back(json(item));
        }
    }
    
    // Universal constructor for any serializable type with fallback
    template<typename T>
    json(const T& obj) {
        // For now, just create a null json for unknown types
        type_ = json_value_type::null;
        static_cast<void>(obj); // avoid unused parameter warning
    }
    
    // Copy constructor
    json(const json& other) {
        type_ = other.type_;
        switch (type_) {
            case json_value_type::object:
                value_.object = new object_t(*other.value_.object);
                break;
            case json_value_type::array:
                value_.array = new array_t(*other.value_.array);
                break;
            case json_value_type::string:
                value_.string = new string_t(*other.value_.string);
                break;
            default:
                value_ = other.value_;
                break;
        }
    }
    
    // Initializer list constructor for object syntax: json{{"key", value}, ...}
    template<typename T>
    json(std::initializer_list<std::pair<const char*, T>> init_list) {
        type_ = json_value_type::object;
        value_.object = new object_t();
        for (const auto& pair : init_list) {
            (*value_.object)[pair.first] = json(pair.second);
        }
    }
    
    json(std::initializer_list<std::pair<std::string, json>> init_list) {
        type_ = json_value_type::object;
        value_.object = new object_t();
        for (const auto& pair : init_list) {
            (*value_.object)[pair.first] = pair.second;
        }
    }
    
    // Destructor
    ~json() {
        clear();
    }
    
    // Assignment operator
    json& operator=(const json& other) {
        if (this != &other) {
            clear();
            type_ = other.type_;
            switch (type_) {
                case json_value_type::object:
                    value_.object = new object_t(*other.value_.object);
                    break;
                case json_value_type::array:
                    value_.array = new array_t(*other.value_.array);
                    break;
                case json_value_type::string:
                    value_.string = new string_t(*other.value_.string);
                    break;
                default:
                    value_ = other.value_;
                    break;
            }
        }
        return *this;
    }
    
    // Template assignment operator for any type
    template<typename T>
    json& operator=(const T& value) {
        *this = json(value);
        return *this;
    }
    
    // Equality comparison with int (for guid comparison)
    bool operator==(int value) const {
        if (type_ == json_value_type::number_integer) {
            return value_.number_integer == value;
        }
        return false;
    }
    
    // General equality comparison
    bool operator==(const json& other) const {
        if (type_ != other.type_) return false;
        
        switch (type_) {
            case json_value_type::null:
                return true;
            case json_value_type::boolean:
                return value_.boolean == other.value_.boolean;
            case json_value_type::number_integer:
                return value_.number_integer == other.value_.number_integer;
            case json_value_type::number_unsigned:
                return value_.number_unsigned == other.value_.number_unsigned;
            case json_value_type::number_float:
                return value_.number_float == other.value_.number_float;
            case json_value_type::string:
                return *value_.string == *other.value_.string;
            case json_value_type::array:
                return *value_.array == *other.value_.array;
            case json_value_type::object:
                return *value_.object == *other.value_.object;
            default:
                return false;
        }
    }
    
    // Object access
    json& operator[](const std::string& key) {
        if (type_ == json_value_type::null) {
            type_ = json_value_type::object;
            value_.object = new object_t();
        }
        if (type_ == json_value_type::object) {
            return (*value_.object)[key];
        }
        static json null_json;
        return null_json;
    }
    
    const json& operator[](const std::string& key) const {
        if (type_ == json_value_type::object) {
            auto it = value_.object->find(key);
            if (it != value_.object->end()) {
                return it->second;
            }
        }
        static json null_json;
        return null_json;
    }
    
    // at() method for checked access
    json& at(const std::string& key) {
        if (type_ == json_value_type::object) {
            auto it = value_.object->find(key);
            if (it != value_.object->end()) {
                return it->second;
            }
        }
        static json null_json;
        return null_json;
    }
    
    const json& at(const std::string& key) const {
        if (type_ == json_value_type::object) {
            auto it = value_.object->find(key);
            if (it != value_.object->end()) {
                return it->second;
            }
        }
        static json null_json;
        return null_json;
    }
    
    // Array access
    json& operator[](std::size_t index) {
        if (type_ == json_value_type::null) {
            type_ = json_value_type::array;
            value_.array = new array_t();
        }
        if (type_ == json_value_type::array) {
            if (index >= value_.array->size()) {
                value_.array->resize(index + 1);
            }
            return (*value_.array)[index];
        }
        static json null_json;
        return null_json;
    }
    
    const json& operator[](std::size_t index) const {
        if (type_ == json_value_type::array && index < value_.array->size()) {
            return (*value_.array)[index];
        }
        static json null_json;
        return null_json;
    }
    
    // Array methods
    void push_back(const json& element) {
        if (type_ == json_value_type::null) {
            type_ = json_value_type::array;
            value_.array = new array_t();
        }
        if (type_ == json_value_type::array) {
            value_.array->push_back(element);
        }
    }
    
    // Iterator support for range-based for loops
    array_t::iterator begin() {
        if (type_ == json_value_type::array) {
            return value_.array->begin();
        }
        static array_t empty_array;
        return empty_array.begin();
    }
    
    array_t::iterator end() {
        if (type_ == json_value_type::array) {
            return value_.array->end();
        }
        static array_t empty_array;
        return empty_array.end();
    }
    
    array_t::const_iterator begin() const {
        if (type_ == json_value_type::array) {
            return value_.array->begin();
        }
        static array_t empty_array;
        return empty_array.begin();
    }
    
    array_t::const_iterator end() const {
        if (type_ == json_value_type::array) {
            return value_.array->end();
        }
        static array_t empty_array;
        return empty_array.end();
    }
    
    // Type checking
    bool is_null() const { return type_ == json_value_type::null; }
    bool is_object() const { return type_ == json_value_type::object; }
    bool is_array() const { return type_ == json_value_type::array; }
    bool is_string() const { return type_ == json_value_type::string; }
    bool is_boolean() const { return type_ == json_value_type::boolean; }
    bool is_number() const { 
        return type_ == json_value_type::number_integer || 
               type_ == json_value_type::number_unsigned || 
               type_ == json_value_type::number_float; 
    }
    
    // Check if object contains key
    bool contains(const std::string& key) const {
        if (type_ == json_value_type::object) {
            return value_.object->find(key) != value_.object->end();
        }
        return false;
    }
    
    // Get size of array or object
    std::size_t size() const {
        switch (type_) {
            case json_value_type::array:
                return value_.array->size();
            case json_value_type::object:
                return value_.object->size();
            case json_value_type::string:
                return value_.string->size();
            default:
                return 0;
        }
    }
    
    // Value conversion
    std::string get() const {
        if (type_ == json_value_type::string) {
            return *value_.string;
        }
        return "";
    }
    
    template<typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return get();
        } else if constexpr (std::is_same_v<T, bool>) {
            return type_ == json_value_type::boolean ? value_.boolean : false;
        } else if constexpr (std::is_integral_v<T>) {
            return type_ == json_value_type::number_integer ? static_cast<T>(value_.number_integer) : T{};
        } else if constexpr (std::is_floating_point_v<T>) {
            return type_ == json_value_type::number_float ? static_cast<T>(value_.number_float) : T{};
        }
        return T{};
    }
    
    // get_to() method for assigning values to references
    template<typename T>
    void get_to(T& val) const {
        val = get<T>();
    }
    
    // Specialization for arrays
    template<typename T, std::size_t N>
    void get_to(T (&arr)[N]) const {
        if (type_ == json_value_type::array && value_.array->size() >= N) {
            for (std::size_t i = 0; i < N; ++i) {
                (*value_.array)[i].get_to(arr[i]);
            }
        }
    }
    
    // Serialization (basic)
    std::string dump(int indent = -1) const {
        (void)indent; // Ignore formatting for now
        switch (type_) {
            case json_value_type::null:
                return "null";
            case json_value_type::string:
                return "\"" + *value_.string + "\"";
            case json_value_type::boolean:
                return value_.boolean ? "true" : "false";
            case json_value_type::number_integer:
                return std::to_string(value_.number_integer);
            case json_value_type::number_unsigned:
                return std::to_string(value_.number_unsigned);
            case json_value_type::number_float:
                return std::to_string(value_.number_float);
            case json_value_type::object:
                return "{}"; // Simplified
            case json_value_type::array:
                return "[]"; // Simplified
        }
        return "null";
    }
    
    // Static parse function (stub)
    static json parse(const std::string& str) {
        (void)str;
        return json(); // Return null JSON for now
    }
    
private:
    void clear() {
        switch (type_) {
            case json_value_type::object:
                delete value_.object;
                break;
            case json_value_type::array:
                delete value_.array;
                break;
            case json_value_type::string:
                delete value_.string;
                break;
            default:
                break;
        }
        type_ = json_value_type::null;
    }
};

// Stream operators
inline std::ostream& operator<<(std::ostream& os, const json& j) {
    return os << j.dump();
}

inline std::istream& operator>>(std::istream& is, json& j) {
    std::string str;
    std::getline(is, str);
    j = json::parse(str);
    return is;
}

// Forward declaration for adl_serializer
template<typename T>
struct adl_serializer {
    static void to_json(json& j, const T& t) {
        static_cast<void>(j);
        static_cast<void>(t);
    }
    
    static void from_json(const json& j, T& t) {
        static_cast<void>(j);
        static_cast<void>(t);
    }
};

} // namespace nlohmann

// Disable problematic macros for compatibility builds
#define NLOHMANN_JSON_SERIALIZE_ENUM(ENUM_TYPE, ...) \
    /* JSON serialization disabled for compatibility */

#define NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(type, ...) \
    /* JSON serialization disabled for compatibility */

#endif /* NLOHMANN_JSON_COMPAT_HPP */
