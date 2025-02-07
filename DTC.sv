/*`timescale 1ns/1fs
module DTC #(parameter integer DIN_WIDTH = 24) (
    input logic sig_in,               // Clock input
    input logic signed [DIN_WIDTH-1:0] d_in, // Digital input to DTC
    output logic sig_out           // Delayed output signal
);
// DTC parameters
    real Tfs = 1.0/(64.5*150e-3);/////0.103359
    real res = Tfs/(2**(DIN_WIDTH -1)); // Time delay per count
    real CM_DELAY = Tfs; //1 / (64.5 * 150e6);
    int  rise_flag,Fall_flag;
    real delay;
always @(posedge sig_in) begin
        rise_flag <= 1;
    end

    always @(negedge sig_in) begin
        Fall_flag <= 1;
    end
    always @(*) begin
        if (rise_flag == 1) begin
            sig_out = 0;
            #delay;
            sig_out = 1'b1;
            rise_flag = 0;
        end
end
always @(*) begin
        if (Fall_flag == 1) begin
            #delay;
            sig_out = 0; 
            Fall_flag = 0; 
        end
        //#delay;
        //sig_out = 0'b1; // Assuming sig_in is always high (you can replace it with sig_in if necessary)
    end
always @(posedge sig_in) begin
        delay =0;//(res * d_in) + CM_DELAY; // Calculate time delay
/*if (d_in==0) delay=0.1222199+CM_DELAY;
else if (d_in==8) delay=-0.032929+CM_DELAY;
else if (d_in==16) delay=0.018805+CM_DELAY;
else delay= 0.070566+CM_DELAY;*/
/*end
endmodule*/


`timescale 1ns/1fs
module DTC #(parameter integer DIN_WIDTH = 10) (
    input logic sig_in,               // Clock input
    input logic signed [DIN_WIDTH-1:0] d_in, // Digital input to DTC
    output logic sig_out           // Delayed output signal
);

    // DTC parameters
    real Tfs = 0.15;//1.0/(64.5*150e-3);/////0.103359
    real KDTC ; //=Tfs/16;
    real res = Tfs/(2**(DIN_WIDTH -1)); // Time delay per count
    real CM_DELAY = Tfs; //1 / (64.5 * 150e6);
    int  rise_flag,Fall_flag;
    real delay,delay_Aps,delay_Aps1;
    always @(posedge sig_in) begin
        rise_flag <= 1;
    end

    always @(negedge sig_in) begin
        Fall_flag <= 1;
    end
    always @(*) begin
        if (rise_flag == 1) begin
            sig_out = 0;
            #delay;
            sig_out = 1'b1;
            rise_flag = 0;
        end
        //#delay;
         // Assuming sig_in is always high (you can replace it with sig_in if necessary)
    end
        always @(*) begin
        if (Fall_flag == 1) begin
            #delay;
            sig_out = 0; 
            Fall_flag = 0; 
        end
    end

    always @(d_in) begin //always @(posedge d_in) begin

      
        KDTC = (0.05/(res*255));//(6.25*0.001)/res;
  	delay_Aps1= res*KDTC*d_in;//(0.01175175*d_in);
        delay = delay_Aps1 + CM_DELAY;
        //delay =6.25*d_in*0.001 + CM_DELAY;//(9.430875*d_in*1e-3) + CM_DELAY;///(res * d_in) + CM_DELAY; // Calculate time delay
    end
endmodule




