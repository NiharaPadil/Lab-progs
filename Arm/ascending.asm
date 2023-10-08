        AREA ASCENDING,CODE,READONLY

ENTRY 
START
        MOV R0,#4
        LDR R1,=CVALUE
        LDR R2,=DVALUE

LOOP    LDR R3,[R1],#4
        STR R3,[R2],#4
        SUB R0,R0,#1
        CMP R0,#0
        BNE LOOP

START1   MOV R4,#3
        MOV R5,#0
        LDR R1,=DVALUE
        
LOOP1   LDR R2,[R1],#4
        LDR R3,[R1]
        CMP R2,R3
        BLS LOOP2
        STR R2,[R1],#-4
        STR R3,[R1]
        MOV R5,#1
        ADD R1,#4

LOOP2   SUB R4,R4,#1
        CMP R4,#0
        BNE LOOP1
        CMP R5,#0
        BNE START1 

B1      B B1


CVALUE  DCD 0X44444444
        DCD 0X11111111
        DCD 0X22222222
        DCD 0X88888888

        AREA DATA2,DATA,READWRITE
DVALUE  DCD 0X00000000
        END
