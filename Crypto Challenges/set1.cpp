#include <cstdint>
#include <iostream>


using std::cout;
using std::endl;

using BYTE = unsigned char;


std::string hex_to_base64(const std::string& in) {
    /* Example
        3 hex bytes         49 27 6d
        to binary           0100 1001 0010 0111 0110 1101
        rearrange to 4x6    010010 010010 011101 101101
        interprete base64   SSdt
    */

	static const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string out;

	unsigned int val = 0, val2 = -6;

	for (BYTE c : in) {  // uchar = byte TODO using or typedef?
		val = (val << 8) + c;  // shift 8 new bits in (byte)
		val2 += 8;  // counter how many bits need to be shifted back for the current 6 bits to process.

		while (val2 >= 0) {
			unsigned int index = (val >> val2) & 0x3F;  // right-most six bits only
			out.push_back(letters[index]);
			val2 -= 6;
		}
	}

    if (val2 > -6) {
		unsigned int index = ((val << 8) >> (val2 + 8)) & 0x3F;  // 0x3F cause we want the right-most six bits
		out.push_back(letters[index]);
	}

    while (out.size() % 4) {
		out.push_back('=');
	}
	
    return out;
}


/*
static std::string base64_decode(const std::string &in) {

    std::string out;

    std::vector<int> T(256,-1);
    for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i; 

    int val=0, valb=-8;
    for (uchar c : in) {
        if (T[c] == -1) break;
        val = (val<<6) + T[c];
        valb += 6;
        if (valb>=0) {
            out.push_back(char((val>>valb)&0xFF));
            valb-=8;
        }
    }
    return out;
}
*/


int main(int argc, char* argv[]) {
 
	std::string hex_string = "49276d206b696c6c696e6720796f757220627261696e206c696b65206120706f69736f6e6f7573206d757368726f6f6d";

	std::string expected_string = "SSdtIGtpbGxpbmcgeW91ciBicmFpbiBsaWtlIGEgcG9pc29ub3VzIG11c2hyb29t";

	cout << hex_to_base64(hex_string) << endl;

    return 0;
}
