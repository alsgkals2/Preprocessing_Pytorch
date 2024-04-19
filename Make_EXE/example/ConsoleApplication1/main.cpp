#include "includePython.h"

#include <iostream>
#include <string>
#include <codecvt>
#include <Windows.h>

using std::wstring;
using std::string;
using namespace std;

int wmain(int argc, wchar_t *argv[]) {
	string path2;
	std::wstring wide_str(argv[0]);
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	path2 = converter.to_bytes(wide_str);
	std::string sliced_str = path2.substr(0, path2.length() - 17);
	cout << sliced_str << endl;
	std::wstring model_pt;
	model_pt.assign(sliced_str.begin(), sliced_str.end());
	sliced_str += "str";
	std::wstring winput_env = L"C:/User/utils/envs/stst";

	cout << "실행 파일 경로: " << path2 << endl;
	wcout << "환경 설정 경로: " << winput_env << endl;
	wcout << endl;
	Py_SetPythonHome(winput_env.c_str());
	Py_Initialize();
	cout << "파일 불러오기" << endl;
	PyObject *custom_module = PyImport_Import(PyUnicode_FromString("Landmark_ceph"));
// <Lateral image path> <ini output path> <Trained model path> <Device>
	if (custom_module != NULL) {
		printf("\nCustom module imported successfully.\n");
		PyObject *process = PyObject_GetAttrString(custom_module, "main");
		if (process) {
			printf("Custom process successfully.\n");
			
			// python example.py arg1 arg2 arg3 arg4
			std::wstring input = argv[1]; // arg1
			std::wstring output = argv[2]; // arg2
			std::wstring weight = argv[3]; // arg3
			std::wstring device = argv[4]; // arg4
			wcout << input << endl;
			PyObject *parameteObj = PyTuple_New(4);
			PyTuple_SetItem(parameteObj, 0, PyUnicode_FromUnicode(input.c_str(), input.size()));
			PyTuple_SetItem(parameteObj, 1, PyUnicode_FromUnicode(output.c_str(), output.size()));
			PyTuple_SetItem(parameteObj, 2, PyUnicode_FromUnicode(weight.c_str(), weight.size()));
			PyTuple_SetItem(parameteObj, 3, PyUnicode_FromUnicode(device.c_str(), device.size()));
			PyObject *r = PyObject_CallObject(process, parameteObj);
		}
		Py_DECREF(custom_module);
	}
	else {
		PyErr_Print();
		PyErr_Print();
		PyErr_Print();
	}


	Py_Finalize();
	return 0;
	system("pause");
}
