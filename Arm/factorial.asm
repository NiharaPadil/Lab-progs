        AREA FACTORIAL,CODE,READONLY
ENTRY
START   
        MOV R0,#5
        MOV R1,R0
        CMP R1,#1
        BLE STOP

FACT    SUB R1,R1,#1
        CMP R1,#1
        BEQ STOP 
        MUL R2,R0,R1
        MOV R0,R2
        BNE FACT
STOP    END
