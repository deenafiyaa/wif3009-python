from audit import ProfileInput, stage1_statistical

test = ProfileInput(
    bio="I dont chase I attract",
    smokes="yes",
    drinks="often",
    drugs="sometimes",
    status="married",
    age=31
)

score, passed = stage1_statistical(test)
print("Risk score:", score)
print("Passed clean:", passed)
print("Result:", "🚩 RED FLAG" if not passed else "✅ CLEAN")