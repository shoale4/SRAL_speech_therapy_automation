%% Code to create figures for poster

xf1 = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/xf1.mat');
xf2 = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/xf2.mat');
xf3 = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/xf3.mat');
xf4 = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/xf4.mat');
xf5 = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/xf5.mat');
fftFirstVowel = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/fftFirstVowel.mat');
fftSecondVowel = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/fftSecondVowel.mat');
fftDifference = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/fftDifference.mat');
fftStudentPlusError = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/fftStudentPlusError.mat');
soundOfDiffs = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/soundOfDiffs.mat');
fft_forAdding = load('/Users/shoale/Documents/Shirley Ryan AbilityLab/Python/fft_forAdding.mat');

xf1 = xf1.xf;
xf2 = xf2.xf;
xf3 = xf3.xf;
xf4 = xf4.xf;
xf5 = xf5.xf;
fftFirstVowel = fftFirstVowel.fftFirstVowel;
fftSecondVowel = fftSecondVowel.fftSecondVowel;
fftDifference = fftDifference.fftDifference;
fftStudentPlusError = fftStudentPlusError.fftStudentPlusError;
soundOfDiffs = soundOfDiffs.soundOfDiffs;
fft_forAdding = fft_forAdding.fft_forAdding;

fontsize = 20;
% figure, plot(xf1, fftFirstVowel), title('Instructor Voice: Frequency Domain','fontsize',fontsize), axis off
% figure, plot(xf2, fftSecondVowel), title('Patient Voice: Frequency Domain','fontsize',fontsize), axis off
% figure, plot(xf3, fftDifference), title('Patient - Instructor (Frequency Domain)','fontsize',fontsize), axis off
% figure, plot(xf4, fftStudentPlusError), title('Error Augmented Sound: Frequency Domain','fontsize',fontsize), axis off
% figure, plot(xf5, soundOfDiffs), title('Error Augmented Sound: Time Domain','fontsize',fontsize), axis off
figure, plot(xf3, fft_forAdding(1:length(fftDifference))), title('Scaled Error (Added to Patient Voice): Frequency Domain','fontsize',fontsize), axis off
