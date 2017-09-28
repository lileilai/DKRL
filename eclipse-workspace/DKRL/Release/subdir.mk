################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Test_cnn.cpp \
../Train_cnn_multi.cpp \
../a.cpp 

OBJS += \
./Test_cnn.o \
./Train_cnn_multi.o \
./a.o 

CPP_DEPS += \
./Test_cnn.d \
./Train_cnn_multi.d \
./a.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


