`timescale 1ns/1ps
module sysDiv (input_clk, N,output_clk,DLF_out);
//parameter integer N;
integer n=26;
input int N;
input input_clk;
output output_clk;
input reg signed[26:0] DLF_out ;
reg signed [26:0] alpha;
wire[6:0] DLF_IN;
wire[17:0] DAC_IN;
wire signed [26:0] QNC ;
//reg signed [26:0] QNC_Delay ; //New comment
wire ClK_Div , DTC_OUT;
real analog_outp,analog_outn,Voutp_max,Voutn_max;
real fvco,tvco_out; //New Paramaters
//wire signed [26:0] QNC_CORR ; //New comment
reg error,rst,frac;

reg signed [9:0] QNC_CORR_10bit;

TDC #(.RESOLUTION (1e-13), .DOUT_WIDTH(2)) tdc (.Clk_ref(input_clk),.feedback(DTC_OUT),.Dout(DLF_IN),.time_diff());
//TDC #(.RESOLUTION (1e-13), .DOUT_WIDTH(7) , .time_unit(1e-15)) tdc (.Clk_ref(input_clk),.feedback(DTC_OUT),.Dout(DLF_IN));
DLF #(.alpha(35.001468),.beta(0.756899),.WIDTH_IN(2),.WIDTH_OUT(18)) dlf (.clk(input_clk),.rst(rst),.IN(DLF_IN),.OUT(DAC_IN));
DAC #(.DATA_WIDTH(18)) dac (.reset(rst),.data(DAC_IN),.analog_outp(analog_outp),.analog_outn(analog_outn),
.max_analog_outp(Voutp_max),.max_analog_outn(Voutn_max));

VCO_new #(.Vmax(1.2),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))vco(.oclk(output_clk),.cfs(6'd26),.N(N),.Voutp(analog_outp)
,.Voutn(analog_outn));

//VCO_system #(.n(24),.Fref(150e6),.Vmax(1.2),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))vco(.N(N),.oclk(output_clk),.Voutp(analog_outp),.Voutn(analog_outn),.alpha(25'd8));



/*VCO_neww #(.vin_max(1.2),.Fref(55e6),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))dut4(.oclk(output_clk),.N(N),.cfs(6'd15),.Voutp(analog_outp)
,.Voutn(analog_outn));*/

clkdiveven #(60) divider(.vco_out(output_clk),.clk_out(ClK_Div),.x(frac));
assign alpha=(27'd16777216)+DLF_out;
Sigma_Delta_internal#(.n(26))sigmadelta(.alpha(alpha), .CLK(ClK_Div) , .out(frac),.QNC(QNC) );

//NEW TDC ADDITIONAL
TDC #(.RESOLUTION (1e-13), .DOUT_WIDTH(7)) freq_mesure (.Clk_ref(output_clk),.feedback(~output_clk),.time_diff(tvco_out),.Dout());
assign fvco = 1.0/(2*tvco_out);

//New comment
/*
assign error=ClK_Div-input_clk;
assign QNC_CORR = int'((QNC_Delay-8)*(-9.825875e-12)*(2**9)/(0.103359e-9));//27.3117808;//;10'd8

DTC #(.DIN_WIDTH(27))dtc (
 .d_in(QNC), // Digital input to DTC
 .sig_in(ClK_Div),   // Input Signal
 .sig_out(DTC_OUT) //  delayed output signal
);
always @(posedge ClK_Div) begin 
QNC_Delay<=QNC;
end
*/

always @(posedge input_clk or posedge DTC_OUT)
begin
    if ((input_clk && !ClK_Div) || (!input_clk && ClK_Div))
        error = input_clk - ClK_Div;
      else error=0;
end

//assign QNC_CORR_10bit= (QNC/(2**(n-5))-8); //Zyad comment

assign QNC_CORR_10bit= (QNC/(2**(n-10))-256);

DTC #(.DIN_WIDTH(10))dtc (
 .d_in(QNC_CORR_10bit), // Digital input to DTC
 .sig_in(ClK_Div),   // Input Signal
 .sig_out(DTC_OUT) //  delayed output signal
);


initial begin rst=1;#0.01 rst=0; end //QNC_Delay=0;end  //New comment  	//alpha=10'd8;
endmodule