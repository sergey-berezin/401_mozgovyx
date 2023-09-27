dotnet sln remove YoloPackage
dotnet remove ConsoleApp reference ./YoloPackage/YoloPackage.csproj
dotnet add ConsoleApp package Mozgovyx.YoloPackage --version 0.1.2309.2
