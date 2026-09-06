# Directory Guide: KGB Archiver 2/paq7

PAQ7 context-mixing algorithm dynamic library project, providing extreme data compression density utilizing advanced context-mixing techniques.

## Files

| File | Type | Description |
| :--- | :--- | :--- |
| [paq7.asm](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7.asm) | Assembly File | Assembly language source file containing highly optimized routine implementations. |
| [paq7.cpp](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7.cpp) | C++ Source | C++ Source code file containing implementation logic. |
| [paq7.rc](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7.rc) | Resource Script | Windows visual compiler resources script definition file. |
| [paq7.vcproj](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7.vcproj) | Project File | Visual Studio VC++ Project configuration file. |
| [paq7asm-x86_64.asm](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7asm-x86_64.asm) | Assembly File | Assembly language source file containing highly optimized routine implementations. |
| [paq7asmsse.asm](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/paq7asmsse.asm) | Assembly File | Assembly language source file containing highly optimized routine implementations. |
| [resource.h](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/resource.h) | Header File | C/C++ Header file containing classes, structures, and function declarations. |
| [stdafx.cpp](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/stdafx.cpp) | C++ Source | C++ Source code file containing implementation logic. |
| [stdafx.h](file:///home/noobcod3r-rtx/Documents/GitHub/KGB-study/KGB%20Archiver%202/paq7/stdafx.h) | Header File | C/C++ Header file containing classes, structures, and function declarations. |


---


  ### 1. Compression
  
  Run from the directory containing your compiled paq7 executable:
  
    ./paq7 -3 enwik9.paq7 /home/noobcod3r-rtx/Documents/GitHub/CUDA-thesis-study/enwik9.txt         
  
  │ Tip
  │
  │ • Memory options range from -1 to -5 (default is -3).
  │ • For faster compression on a 1 GB file like enwik9.txt, use -1 (uses ~62 MB RAM).              
  │ • For highest compression ratio, use -5 (uses ~525 MB RAM).
  ──────
  ### 2. Decompression
  
  #### Extract to a new / specified file:
  
  Because PAQ7 does not overwrite existing files by default, specify the destination file name:     
  
    ./paq7 enwik9.paq7 /home/noobcod3r-rtx/Documents/GitHub/CUDA-thesis-study/enwik9_extracted.txt  
  
  #### Extract using the stored filename:
  
  If the original file does not exist at its original path:
  
    ./paq7 enwik9.paq7
  
  #### Verify / Compare against archive:
  
  If enwik9.txt already exists at the target path, running:
  
    ./paq7 enwik9.paq7 /home/noobcod3r-rtx/Documents/GitHub/CUDA-thesis-study/enwik9.txt            
  
  will compare the archive against the file and report if they are identical.
